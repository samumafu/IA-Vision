"""Compare a Transformer and CNN fusion policies on the fixed development sets."""

import argparse
import csv
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from app.detector import NeuralDetector, crop_box, primary_detection
from app.vision import InferenceService, decode_image
from scripts.diagnose_scene_detector import load_scene, validation_rows
from scripts.evaluate_detector import MATERIALS
from scripts.evaluate_scene_materials import image_from, predict
from scripts.train_scene_materials import ORIGINAL_PREFIX, digest, original_rows, scores
from training.detection_metrics import match
from training.prediction import protect_test_image
from utils.config import CLASSES


VIEWS_KEY = '512_512flip_lone0.78'


def policies(probabilities, scene_weight, calibrations=None):
    refined, scene = probabilities['refined'], probabilities['scene']
    active = (1 - scene_weight) * refined + scene_weight * scene
    result = {'refined': refined, 'active': active}
    top_two = np.sort(active, axis=1)[:, -2:]
    margins = top_two[:, 1] - top_two[:, 0]
    target_indices = [CLASSES.index(label) for label in ('plastico', 'papel', 'carton', 'organicos')]
    base_winner = active.argmax(1)
    target_baseline = np.isin(base_winner, target_indices)
    for name, transformer in probabilities.items():
        if name in ('refined', 'scene'):
            continue
        result[name] = transformer
        for weight in (.15, .25, .40, .55, .70):
            result[f'{name}_replace_{weight:g}'] = (1 - weight) * refined + weight * transformer
            result[f'{name}_blend_{weight:g}'] = (1 - weight) * active + weight * transformer
        for weight in (.30, .45, .60):
            result[f'{name}_with_scene_{weight:g}'] = (1 - weight) * scene + weight * transformer
        for maximum_margin in (.15, .28, .45):
            mask = (margins < maximum_margin)[:, None]
            result[f'{name}_guard_{maximum_margin:g}'] = np.where(mask, .45 * active + .55 * transformer, active)
        focus = target_baseline & np.isin(transformer.argmax(1), target_indices) & (margins < .35)
        result[f'{name}_focus'] = np.where(focus[:, None], .45 * active + .55 * transformer, active)
        paper_safe = focus & (base_winner != CLASSES.index('papel'))
        result[f'{name}_focus_paper_safe'] = np.where(paper_safe[:, None], .45 * active + .55 * transformer, active)
        protected = paper_safe & (base_winner != CLASSES.index('carton'))
        result[f'{name}_focus_protected'] = np.where(protected[:, None], .45 * active + .55 * transformer, active)
        for variant, calibration in (calibrations or {}).get(name, {}).items():
            logits = np.log(np.clip(np.concatenate((active, transformer), axis=1), 1e-6, 1)) @ calibration['matrix']
            logits += calibration['bias']
            calibrated = np.exp(logits - logits.max(axis=1, keepdims=True))
            calibrated /= calibrated.sum(axis=1, keepdims=True)
            key = f'{name}_calibrated' + (f'_{variant}' if variant else '')
            result[key] = calibrated
            result[f'{key}_blend'] = .35 * active + .65 * calibrated
            keep_cnn = np.isin(base_winner, [CLASSES.index('papel'), CLASSES.index('carton')])[:, None]
            result[f'{key}_protected'] = np.where(keep_cnn, active, calibrated)
            result[f'{key}_blend_protected'] = np.where(keep_cnn, active, result[f'{key}_blend'])
            blended_winner = result[f'{key}_blend'].argmax(1)
            evidence = transformer[np.arange(len(transformer)), blended_winner]
            needs_review = keep_cnn[:, 0] & (blended_winner != base_winner)
            keep = needs_review & ((evidence < .60) | (transformer.argmax(1) != blended_winner))
            result[f'{key}_review'] = np.where(keep[:, None], active, result[f'{key}_blend'])
            paper_review = keep | (base_winner == CLASSES.index('papel'))
            result[f'{key}_review_keep_paper'] = np.where(paper_review[:, None], active, result[f'{key}_blend'])
            keep_strict = needs_review & ((evidence < .80) | (transformer.argmax(1) != blended_winner))
            result[f'{key}_review_strict'] = np.where(keep_strict[:, None], active, result[f'{key}_blend'])
            glass = base_winner == CLASSES.index('vidrio')
            paper = base_winner == CLASSES.index('papel')
            result[f'{key}_preserve_glass'] = np.where(glass[:, None], active, result[f'{key}_blend'])
            result[f'{key}_preserve_glass_paper'] = np.where((glass | paper)[:, None], active, result[f'{key}_blend'])
            fibre = paper | (base_winner == CLASSES.index('carton'))
            consensus = (refined.argmax(1) == base_winner) & (scene.argmax(1) == base_winner)
            result[f'{key}_preserve_consensus'] = np.where((glass | (fibre & consensus))[:, None], active, result[f'{key}_blend'])
    return result


def class_scores(probabilities, labels):
    truth = [CLASSES.index(label) for label in labels]
    return {key: scores(truth, value.argmax(1)) for key, value in probabilities.items()}


def material_crops(records, root, threshold):
    scenes = {r['id']: r for r in validation_rows(root)}
    if len(records) != len(scenes) or {r['image_id'] for r in records} != set(scenes):
        raise ValueError('D3 predictions must cover the entire fixed validation partition exactly once.')
    denominator = sum(t['category'] in MATERIALS for r in records for t in r['truth'])
    pictures, labels, primary_indices = [], [], []
    for record in records:
        predictions = [p for p in record['predictions'] if p['score'] >= threshold]
        pairs = [(i, j) for i, j, _ in match(predictions, record['truth'])
                 if record['truth'][j]['category'] in MATERIALS]
        if not pairs:
            continue
        rgb = load_scene(root, scenes[record['image_id']])
        h, w = rgb.shape[:2]
        for i, j in pairs:
            if i == 0:
                primary_indices.append(len(pictures))
            x1, y1, x2, y2 = crop_box(predictions[i]['box'], w, h)
            pictures.append(Image.fromarray(rgb[y1:y2, x1:x2]))
            labels.append(MATERIALS[record['truth'][j]['category']])
    return pictures, labels, primary_indices, denominator


def evaluate(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    checkpoints = dict(value.split('=', 1) for value in args.transformers)
    if len(checkpoints) != len(args.transformers) or (set(checkpoints) & {'refined', 'scene', 'active'}):
        raise ValueError('Candidate names must be distinct from the CNN reference names.')
    models = {name: InferenceService(path) for name, path in
              ({'refined': config['checkpoint'], 'scene': config['material_candidate_checkpoint']} | checkpoints).items()}
    calibrations = {}
    for value in args.fusions:
        identity, path = value.split('=', 1)
        name, _, variant = identity.partition(':')
        calibration = json.loads(Path(path).read_text(encoding='utf-8'))
        required = {'refined': models['refined'].checkpoint_sha256, 'scene': models['scene'].checkpoint_sha256,
                    'transformer': models[name].checkpoint_sha256}
        if (calibration['checkpoints'] != required or calibration['classes'] != list(CLASSES) or calibration['test_used'] is not False or
                calibration.get('scene_weight', config['material_candidate_weight']) != config['material_candidate_weight']):
            raise ValueError('Calibration does not match the selected own checkpoints.')
        calibrations.setdefault(name, {})[variant] = calibration

    def compute(pictures):
        return policies(predict(models, pictures), config['material_candidate_weight'], calibrations)

    results = {'test_used': False, 'checkpoints': {name: model.checkpoint_sha256 for name, model in models.items()}}
    with zipfile.ZipFile(args.original) as archive:
        pictures, labels = [], []
        for row in original_rows(archive):
            if row['split'] != 'val':
                continue
            content = archive.read(ORIGINAL_PREFIX + row['path'])
            if digest(content) != row['sha256']:
                raise ValueError('Original validation image changed.')
            pictures.append(image_from(content))
            labels.append(row['class'])
        results['original'] = class_scores(compute(pictures), labels)

    rows = [r for r in json.loads((args.legacy/'manifest.json').read_text()) if r['split'] == 'val']
    results['TACO_legacy'] = class_scores(compute([image_from((args.legacy/r['path']).read_bytes()) for r in rows]),
                                         [r['class'] for r in rows])
    with zipfile.ZipFile(args.scenes) as archive:
        rows = [r for r in json.loads(archive.read('manifest.json')) if r['split'] == 'val']
        results['TACO_strict'] = class_scores(compute([image_from(archive.read(r['paths']['square'])) for r in rows]),
                                             [r['class'] for r in rows])

    detector = NeuralDetector(config['detector_checkpoint'], confidence=config['detection_threshold'],
                              nms_iou=config['detection_nms_iou'], views=config['detection_views'])
    detection_metrics = json.loads(args.detection_metrics.read_text(encoding='utf-8'))
    if (detection_metrics['checkpoint_sha256'] != detector.checkpoint_sha256 or
            config['detection_views'] != [[512, False], [512, True]] or config['detection_nms_iou'] != .45):
        raise ValueError('Saved detections do not correspond to the active D3 configuration.')
    detections = json.loads(args.predictions.read_text())[VIEWS_KEY]
    pictures, labels, primary, total = material_crops(detections, args.detection_root, config['detection_threshold'])
    truth = np.array([CLASSES.index(label) for label in labels])
    results['joint'] = {}
    for key, probability in compute(pictures).items():
        correct = probability.argmax(1) == truth
        results['joint'][key] = {'correct': int(correct.sum()), 'primary_correct': int(correct[primary].sum()),
                                 'recall': float(correct.sum() / total), 'primary_recall': float(correct[primary].sum() / total),
                                 'matched': len(truth), 'objects': total}

    field_samples = []
    with args.field_labels.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            path = Path('data/field_samples/samples')/row['filename']
            if not path.is_file():
                continue
            protect_test_image(path)
            rgb = decode_image(path.read_bytes())
            proposal = primary_detection(detector.detect(rgb))
            if proposal is not None:
                x1, y1, x2, y2 = proposal.region.box
                field_samples.append((row, Image.fromarray(rgb[y1:y2, x1:x2])))
    results['field'] = []
    if field_samples:
        labels_by_policy = {name: value.argmax(1) for name, value in compute([sample for _, sample in field_samples]).items()}
        results['field'] = [row | {'predictions': {name: CLASSES[values[i]] for name, values in labels_by_policy.items()}}
                            for i, (row, _) in enumerate(field_samples)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Validation saved to {args.output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('config/app.json'))
    p.add_argument('--transformers', nargs='+', required=True)
    p.add_argument('--fusions', nargs='*', default=[])
    p.add_argument('--original', type=Path, default=Path('exports/sync/B/Vision-IA_Colab_B.zip'))
    p.add_argument('--legacy', type=Path, default=Path('data/supplemental/material_crops'))
    p.add_argument('--scenes', type=Path, default=Path('runs/scene_quality/scene_materials.zip'))
    p.add_argument('--detection-root', type=Path, default=Path('data/detection/refine_D2'))
    p.add_argument('--predictions', type=Path, default=Path('runs/intensive/d3_final_fused/predictions.json'))
    p.add_argument('--detection-metrics', type=Path, default=Path('runs/intensive/d3_final/metrics.json'))
    p.add_argument('--field-labels', type=Path, default=Path('config/field_labels_initial.csv'))
    p.add_argument('--output', type=Path, required=True)
    evaluate(p.parse_args())

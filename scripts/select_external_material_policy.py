"""Score prospective runtime policies on existing validation sets before external evaluation."""
import argparse
from itertools import chain
import json
from pathlib import Path
import zipfile

import numpy as np

from scripts.evaluate_material_transformer import VIEWS_KEY, material_crops, policies
from scripts.evaluate_scene_materials import image_from, predict
from scripts.train_scene_materials import ORIGINAL_PREFIX, digest, original_rows, scores
from app.vision import InferenceService
from utils.config import CLASSES


def verified_picture(row, content):
    if digest(content) != row['sha256']:
        raise ValueError(f"Validation photo or crop changed: {row['path']}")
    return image_from(content)


def datasets():
    with zipfile.ZipFile('exports/sync/B/Vision-IA_Colab_B.zip') as source:
        rows = [r for r in original_rows(source) if r['split'] == 'val']
        yield 'original', rows, [verified_picture(r, source.read(ORIGINAL_PREFIX + r['path'])) for r in rows]
    taco = Path('data/supplemental/material_crops')
    rows = [r for r in json.loads((taco / 'manifest.json').read_text(encoding='utf-8')) if r['split'] == 'val']
    yield 'TACO', rows, [verified_picture(r, (taco / r['path']).read_bytes()) for r in rows]
    with zipfile.ZipFile('runs/near_camera/mmorwad-materials.zip') as source:
        rows = [r for r in json.loads(source.read('manifest.json')) if r['split'] == 'val']
        yield 'MMoRWaD', rows, [image_from(source.read(r['paths']['square'])) for r in rows]


def fixed_detector_crops(threshold):
    data = Path('runs/intensive/d3_final_fused/predictions.json')
    records = json.loads(data.read_text(encoding='utf-8'))[VIEWS_KEY]
    return material_crops(records, Path('data/detection/refine_D2'), threshold)


def evaluate(output, candidate, calibration, active_config, transformer_config=None, keep_fiber_guard=False):
    active = json.loads(active_config.read_text(encoding='utf-8'))
    models = {name: InferenceService(file) for name, file in {
        'refined': active['checkpoint'], 'scene': active['material_candidate_checkpoint'],
        'current': active['material_refiner_checkpoint'], 'candidate': candidate,
    }.items()}
    live_cal = json.loads(Path(active['material_refiner_calibration']).read_text(encoding='utf-8'))
    candidate_cal = json.loads(calibration.read_text(encoding='utf-8'))
    calibrations = {'current': {'live': live_cal}, 'candidate': {'new': candidate_cal}}
    configurations = {'active': active}
    standalone = {key: value for key, value in active.items() if not key.startswith(('material_candidate', 'material_refiner'))}
    configurations['standalone'] = standalone | {'checkpoint': candidate.as_posix()}
    transformer_settings = None
    if transformer_config is not None:
        transformer_settings = json.loads(transformer_config.read_text(encoding='utf-8'))
        transformer_calibration = json.loads(Path(transformer_settings['material_refiner_calibration']).read_text(encoding='utf-8'))
        expected = {key: models[name].checkpoint_sha256 for key, name in
                    (('refined', 'refined'), ('scene', 'candidate'), ('transformer', 'current'))}
        if (transformer_calibration['checkpoints'] != expected or
                transformer_calibration['scene_weight'] != transformer_settings['material_candidate_weight'] or
                transformer_settings.get('material_refiner_preserve') or
                transformer_settings.get('material_refiner_preserve_consensus')):
            raise ValueError('The retained Transformer must use the calibrated no-guard configuration.')
        for weight in (.35, .65, 1.):
            configurations[f'cnn_transformer_{weight:g}'] = transformer_settings | {'material_refiner_weight': weight}
            if keep_fiber_guard:
                configurations[f'cnn_transformer_consensus_{weight:g}'] = configurations[f'cnn_transformer_{weight:g}'] | {
                    'material_refiner_preserve': active['material_refiner_preserve'],
                    'material_refiner_preserve_consensus': active['material_refiner_preserve_consensus'],
                }
    for weight in (.35, .65, 1.):
        for guard in ('plain', 'consensus'):
            cfg = active | {'material_refiner_checkpoint': candidate.as_posix(),
                            'material_refiner_weight': weight,
                            'material_refiner_calibration': calibration.as_posix()}
            if guard == 'plain':
                cfg.update(material_refiner_preserve=[], material_refiner_preserve_consensus=[])
            configurations[f'{guard}_{weight:g}'] = cfg
    results = {}
    pictures, labels, primary, total = fixed_detector_crops(active['detection_threshold'])
    joint_rows = [{'class': label} for label in labels]
    for name, rows, pictures in chain(datasets(), [('fixed_D3', joint_rows, pictures)]):
        q = predict(models, pictures)
        calculated = policies(q, active['material_candidate_weight'], calibrations)
        variants = {'active': calculated['current_calibrated_live_preserve_consensus'],
                    'standalone': q['candidate']}
        cnn = calculated['active']
        calibrated = calculated['candidate_calibrated_new']
        baseline = cnn.argmax(1)
        agreed = (q['refined'].argmax(1) == baseline) & (q['scene'].argmax(1) == baseline)
        preserved = (baseline == CLASSES.index('vidrio')) | (agreed & (
            (baseline == CLASSES.index('papel')) | (baseline == CLASSES.index('carton'))))
        for weight in (.35, .65, 1.):
            blended = (1. - weight) * cnn + weight * calibrated
            variants[f'plain_{weight:g}'] = blended
            guarded = blended.copy()
            guarded[preserved] = cnn[preserved]
            variants[f'consensus_{weight:g}'] = guarded
        if transformer_settings is not None:
            new_cnn = (1 - transformer_settings['material_candidate_weight']) * q['refined']
            new_cnn += transformer_settings['material_candidate_weight'] * q['candidate']
            new = policies({'refined': q['refined'], 'scene': q['candidate'], 'current': q['current']},
                           transformer_settings['material_candidate_weight'], {'current': {'new': transformer_calibration}})
            learned = new['current_calibrated_new']
            guarded_classes = [CLASSES.index(label) for label in active['material_refiner_preserve']]
            consensus_classes = [CLASSES.index(label) for label in active['material_refiner_preserve_consensus']]
            new_winner = new_cnn.argmax(1)
            new_agreement = (q['refined'].argmax(1) == new_winner) & (q['candidate'].argmax(1) == new_winner)
            new_preserved = np.isin(new_winner, guarded_classes) | (new_agreement & np.isin(new_winner, consensus_classes))
            for weight in (.35, .65, 1.):
                variants[f'cnn_transformer_{weight:g}'] = (1 - weight) * new_cnn + weight * learned
                if keep_fiber_guard:
                    guarded = variants[f'cnn_transformer_{weight:g}'].copy()
                    guarded[new_preserved] = new_cnn[new_preserved]
                    variants[f'cnn_transformer_consensus_{weight:g}'] = guarded
        truth = [CLASSES.index(row['class']) for row in rows]
        results[name] = {key: scores(truth, probabilities.argmax(1)) for key, probabilities in variants.items()}
        if name == 'fixed_D3':
            expected = np.asarray(truth)
            for key, probabilities in variants.items():
                correct = probabilities.argmax(1) == expected
                results[name][key].update(joint_recall=float(correct.sum() / total),
                                          joint_primary_recall=float(correct[primary].sum() / total),
                                          joint_denominator=total)
    taco, original, near = (results[key] for key in ('TACO', 'original', 'MMoRWaD'))
    valid = [key for key in configurations if key.startswith('cnn_transformer_') and
             taco[key]['accuracy'] >= taco['active']['accuracy'] and
             original[key]['accuracy'] >= original['active']['accuracy'] - .025]
    if keep_fiber_guard:
        valid = [key for key in valid if taco[key]['per_class_recall']['papel'] >= taco['active']['per_class_recall']['papel']]
    ranked = sorted((key for key in configurations if key != 'active'),
                    key=lambda key: (.50 * taco[key]['macro_F1_present'] + .35 * near[key]['macro_F1_present']
                                     + .15 * original[key]['macro_F1']), reverse=True)
    selected = next((key for key in ranked if key in valid), None)
    trial = next((key for key in ranked if key.startswith('cnn_transformer_')), None)
    output.mkdir(parents=True, exist_ok=True)
    shortlisted = selected or trial
    shortlist = ('standalone', shortlisted) if shortlisted else ('standalone',)
    for name in shortlist:
        (output / f'{name}.json').write_text(json.dumps(configurations[name], indent=2), encoding='utf-8')
    (output / 'validation.json').write_text(json.dumps({'results': results, 'selected': selected, 'trial': trial,
                                                        'ranked': ranked}, indent=2),
                                             encoding='utf-8')
    print(output, 'selected on existing validation:', selected, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('calibration', type=Path)
    parser.add_argument('--active', type=Path, default=Path('config/app.json'))
    parser.add_argument('--transformer-config', type=Path)
    parser.add_argument('--keep-fiber-guard', action='store_true')
    args = parser.parse_args()
    evaluate(args.output, args.candidate, args.calibration, args.active, args.transformer_config, args.keep_fiber_guard)

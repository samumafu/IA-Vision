"""Compare checkpoints, crop geometry and guard policies on fixed validation inputs."""
import argparse
from collections import defaultdict
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from app.detector import crop_box
from app.vision import InferenceService, Region
from scripts.diagnose_scene_detector import load_scene, validation_rows
from scripts.evaluate_detector import MATERIALS as LEGACY_LABELS
from scripts.train_scene_materials import ORIGINAL_PREFIX, original_rows, scores
from training.detection_metrics import match
from training.prediction import model_probabilities
from utils.config import CLASSES


@torch.inference_mode()
def predict(models, pictures):
    results = {}
    for key, service in models.items():
        batches = []
        for start in range(0, len(pictures), 32):
            tensors = torch.stack([service.transform(im) for im in pictures[start:start + 32]])
            batches.append(model_probabilities(service.model, tensors, service.metadata['config']).numpy())
        results[key] = np.concatenate(batches) if batches else np.empty((0, len(CLASSES)))
    return results


def guard(probabilities, candidate, cutoff):
    confidence = probabilities[candidate].max(axis=1)
    return np.where((confidence >= cutoff)[:, None], probabilities[candidate], probabilities['B'])


def policies(probabilities):
    guards = {f'guard_{t:.2f}': guard(probabilities, 'refined', t) for t in (.30, .35, .40, .45, .50, .60, .70)}
    blends = {}
    for candidate in probabilities:
        if candidate in ('B', 'refined', 'C1b'):
            continue
        for weight in (.35, .45, .55, .65, .75):
            blends[f'{candidate}_blend_{weight:g}'] = weight * probabilities[candidate] + (1 - weight) * probabilities['refined']
    return probabilities | guards | blends


def image_from(content):
    with Image.open(BytesIO(content)) as picture:
        return ImageOps.exif_transpose(picture).convert('RGB')


def compare(models, pictures, rows):
    probabilities = policies(predict(models, pictures))
    truth = np.array([CLASSES.index(row['class']) for row in rows])
    categories = np.array([row.get('category', row.get('dataset', 'unknown')) for row in rows])
    metrics, decisions = {}, {}
    for key, probability in probabilities.items():
        predicted = probability.argmax(axis=1)
        metrics[key] = scores(truth, predicted)
        metrics[key]['by_source_category'] = {c: {'support': int((categories == c).sum()),
                                                 'correct': int((predicted[categories == c] == truth[categories == c]).sum())}
                                             for c in sorted(set(categories))}
        decisions[key] = predicted
    return metrics, decisions


def labeled_crops(models, root, detector_runs, scene_rows, settings):
    allowed = {(r['image_id'], r['id']): r['class'] for r in scene_rows}
    scenes = {r['id']: r for r in validation_rows(root)}
    threshold, view, primary_only = settings
    pictures, examples = [], []
    denominators = defaultdict(int)
    for record in detector_runs:
        image_id = record['image_id']
        selected = [p for p in record['predictions'] if p['score'] >= threshold]
        if primary_only:
            selected = selected[:1]
        for t in record['truth']:
            for metric, label in (('legacy', LEGACY_LABELS.get(t['category'])), ('strict', allowed.get((image_id, t['id'])))):
                denominators[metric] += bool(label)
        pairs = [(i, j) for i, j, _ in match(selected, record['truth'])
                 if record['truth'][j]['category'] in LEGACY_LABELS or (image_id, record['truth'][j]['id']) in allowed]
        if not pairs:
            continue
        rgb = load_scene(root, scenes[image_id])
        h, w = rgb.shape[:2]
        for i, j in pairs:
            truth = record['truth'][j]
            for kind, box in (('detected', selected[i]['box']), ('oracle', truth['box'])):
                region = crop_box(box, w, h, 0. if view == 'tight' else .12, square=view == 'square')
                x1, y1, x2, y2 = region
                pictures.append(Image.fromarray(rgb[y1:y2, x1:x2]))
            examples.append((LEGACY_LABELS.get(truth['category']), allowed.get((image_id, truth['id']))))
    all_scores = policies(predict(models, pictures))
    output = {}
    for index, metric in enumerate(('legacy', 'strict')):
        indices = np.array([i for i, e in enumerate(examples) if e[index]], dtype=np.intp)
        truth = np.array([CLASSES.index(examples[i][index]) for i in indices])
        output[metric] = {}
        for name, probability in all_scores.items():
            detected, oracle = probability.argmax(axis=1).reshape(-1, 2)[indices].T
            good = int((detected == truth).sum())
            output[metric][name] = {'known_objects': denominators[metric], 'matched': len(truth), 'correct': good,
                                    'joint_recall': good / max(denominators[metric], 1), 'detected': scores(truth, detected),
                                    'same_objects_oracle': scores(truth, oracle)}
    return output


def make_contact_sheet(path, rows, pictures, predictions):
    positions = defaultdict(list)
    for i, row in enumerate(rows):
        actual = CLASSES.index(row['class'])
        if predictions['guard_0.40'][i] != actual:
            positions[row['class']].append(i)
    indices = [i for category in CLASSES for i in positions[category][:4]]
    if not indices:
        return
    page = Image.new('RGB', (4 * 210, ((len(indices) + 3) // 4) * 150), 'white')
    draw = ImageDraw.Draw(page)
    for j, i in enumerate(indices):
        left, top = (j % 4) * 210, (j // 4) * 150
        page.paste(ImageOps.contain(pictures[i], (198, 102)), (left + 5, top + 4))
        draw.text((left + 5, top + 108), f"{rows[i]['image_id']}/{rows[i]['id']} {rows[i]['category']}", fill='black')
        draw.text((left + 5, top + 126), f"{rows[i]['class']} -> {CLASSES[predictions['guard_0.40'][i]]}", fill='black')
    page.save(path, quality=92)


def review_mju(models, root, labels_path):
    with labels_path.open(encoding='utf-8', newline='') as stream:
        labels = list(csv.DictReader(stream))
    rows = {r['id']: r for r in validation_rows(root) if r.get('source_dataset') == 'MJU-Waste'}
    pictures = []
    for label in labels:
        row = rows[int(label['image_id'])]
        annotation = next(b for b in row['boxes'] if b['annotation_id'] == int(label['annotation_id']))
        rgb = load_scene(root, row)
        h, w = rgb.shape[:2]
        box = [v * s for v, s in zip(annotation['xyxy_normalized'], (w, h, w, h))]
        x1, y1, x2, y2 = crop_box(box, w, h)
        pictures.append(Image.fromarray(rgb[y1:y2, x1:x2]))
    return compare(models, pictures, labels)[0]


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    paths = dict(item.split('=', 1) for item in args.models)
    models = {key: InferenceService(path) for key, path in paths.items()}
    with zipfile.ZipFile(args.original) as archive:
        originals = [r for r in original_rows(archive) if r['split'] == 'val']
        pictures = []
        for row in originals:
            raw = archive.read(ORIGINAL_PREFIX + row['path'])
            if hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError(f"Changed original: {row['path']}")
            pictures.append(image_from(raw))
        original = compare(models, pictures, originals)[0]
    legacy_rows = [r for r in json.loads((args.legacy / 'manifest.json').read_text()) if r['split'] == 'val']
    legacy_pictures = [image_from((args.legacy / r['path']).read_bytes()) for r in legacy_rows]
    legacy = compare(models, legacy_pictures, legacy_rows)[0]
    with zipfile.ZipFile(args.scenes) as archive:
        rows = [r for r in json.loads(archive.read('manifest.json')) if r['split'] == 'val']
        variants = {}
        for view in ('square', 'rect', 'tight', 'foreground', 'background'):
            pictures = [image_from(archive.read(r['paths'][view])) for r in rows]
            variants[view], predictions = compare(models, pictures, rows)
            if view == 'square':
                make_contact_sheet(args.output / 'classifier_errors.jpg', rows, pictures, predictions)
                samples = [np.asarray(image) for image in pictures if min(image.size) >= 16][:55]
    timing = defaultdict(list)
    guard_service = InferenceService(paths['B'], candidate_checkpoint=paths['refined'], candidate_threshold=.4)
    for name, service in (models | {'guard_0.40': guard_service}).items():
        for sample in samples:
            start = time.perf_counter()
            service.classify(sample, region=Region((0, 0, sample.shape[1], sample.shape[0]), 'benchmark'))
            timing[name].append((time.perf_counter() - start) * 1000)
    timing = {name: {'median': float(np.median(values[5:])), 'p95': float(np.percentile(values[5:], 95))} for name, values in timing.items()}
    report = {'class_order': CLASSES, 'original': original, 'legacy_308': legacy, 'strict_views': variants,
              'cpu_ms': timing, 'checkpoints': {k: s.checkpoint_sha256 for k, s in models.items()}, 'test_opened': False}
    if args.mju_labels:
        report['independent_MJU_manual_material'] = review_mju(models, args.scene_root, args.mju_labels)
    summary = args.output / 'metrics.json'
    summary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.detector:
        detections = json.loads(args.detector.read_text(encoding='utf-8'))
        report['joint'] = {}
        for detector, threshold in args.operating:
            geometries = [(view, False) for view in args.joint_geometry] + [('square', True)]
            for view, primary in geometries:
                key = f'{detector}:{threshold}:{view}' + (':primary' if primary else '')
                report['joint'][key] = labeled_crops(models, args.scene_root, detections[detector], rows, (float(threshold), view, primary))
                summary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {summary}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, default=Path('exports/sync/B/Vision-IA_Colab_B.zip'))
    p.add_argument('--legacy', type=Path, default=Path('data/supplemental/material_crops'))
    p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--scene-root', type=Path, default=Path('data/detection/refine_D2'))
    p.add_argument('--mju-labels', type=Path)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--detector', type=Path)
    p.add_argument('--operating', action='append', nargs=2, default=[])
    p.add_argument('--joint-geometry', nargs='+', choices=('square', 'rect', 'tight'), default=['square', 'rect', 'tight'])
    p.add_argument('--output', type=Path, required=True)
    evaluate(p.parse_args())

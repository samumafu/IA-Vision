"""Compare D2 operating points on the fixed validation scenes (never test)."""
import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from app.detector import NeuralDetector
from training.detection_metrics import box_iou, match, nms, spatial_metrics


def load_scene(root, row):
    content = (root / 'images' / row['split'] / f"{row['id']:05d}.jpg").read_bytes()
    if hashlib.sha256(content).hexdigest() != row['sha256']:
        raise ValueError(f"Modified photo: {row['id']}")
    with Image.open(BytesIO(content)) as image:
        return np.asarray(image.convert('RGB'))


def validation_rows(root):
    rows = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    splits = {split: [r for r in rows if r['split'] == split] for split in ('train', 'val', 'test')}
    for key in ('id', 'group', 'sha256', 'pixels_sha256'):
        for first, second in combinations(splits, 2):
            if {r[key] for r in splits[first]} & {r[key] for r in splits[second]}:
                raise ValueError(f'{first}/{second} overlap on {key}')
    return splits['val']


def operating_points(records):
    ranking = spatial_metrics(records, .25)
    total = ranking['objects']
    points = {}
    for threshold in (.06, .08, .10, .12, .15, .18, .20, .23, .25, .26, .27, .275, .28, .29, .30, .35):
        predictions = [[p for p in r['predictions'] if p['score'] >= threshold] for r in records]
        tp = sum(len(match(p, r['truth'])) for p, r in zip(predictions, records))
        count = sum(map(len, predictions))
        points[str(threshold)] = {'precision': tp / max(count, 1), 'recall': tp / total,
                                  'TP': tp, 'FP': count - tp, 'FN': total - tp}
    return {key: ranking[key] for key in ('images', 'objects', 'mAP50', 'mAP50_95')} | {'points': points}


def miss_analysis(records):
    groups = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    causes = defaultdict(int)
    for row in records:
        truth = row['truth']
        operating = [p for p in row['predictions'] if p['score'] >= .25]
        found = {j for _, j, _ in match(operating, truth)}
        width, height = row['size']
        scale = 512 / max(width, height)
        for j, item in enumerate(truth):
            x1, y1, x2, y2 = item['box']
            side = ((x2 - x1) * (y2 - y1)) ** .5 * scale
            overlap = max((box_iou(item['box'], other['box']) for k, other in enumerate(truth) if k != j), default=0)
            groups_for_object = {'size_512': '<16' if side < 16 else '16–32' if side < 32 else '32–64' if side < 64 else '>=64',
                                 'category': item['category'], 'source': row['source'],
                                 'overlap_proxy': 'IoU>=.1' if overlap >= .1 else 'IoU<.1',
                                 'crop_luma': '<70' if item['luma'] < 70 else '70–170' if item['luma'] < 170 else '>=170',
                                 'surrounding_texture': 'high' if item['background_texture'] >= 18 else 'low'}
            for axis, label in groups_for_object.items():
                counts = groups[axis][label]
                counts[0] += 1
                counts[1] += j in found
            if j not in found:
                best = max((box_iou(item['box'], p['box']) for p in row['predictions']), default=0)
                has_box = any(box_iou(item['box'], p['box']) >= .5 for p in operating)
                cause = 'competing_match' if has_box else 'below_threshold' if best >= .5 else 'IoU_.1_to_.5' if best >= .1 else 'no_box_IoU_.1'
                causes[cause] += 1
    return {'counts_total_and_detected': groups, 'miss_causes_after_NMS': causes}


def describe(rgb, box):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    left, top = max(0, int(x1)), max(0, int(y1))
    right, bottom = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    dx, dy = max(2, (right - left) // 4), max(2, (bottom - top) // 4)
    outer_left, outer_top = max(0, left - dx), max(0, top - dy)
    gray = cv2.cvtColor(rgb[outer_top:min(h, bottom + dy), outer_left:min(w, right + dx)], cv2.COLOR_RGB2GRAY)
    inside = np.zeros(gray.shape, dtype=bool)
    inside[top-outer_top:bottom-outer_top, left-outer_left:right-outer_left] = True
    gradients = cv2.Laplacian(gray, cv2.CV_32F)
    return (float(gray[inside].mean()) if inside.any() else 0.,
            float(gradients[~inside].std()) if (~inside).any() else 0.)


def tile_predictions(detector, rgb, fraction=.64):
    height, width = rgb.shape[:2]
    tw, th = round(width * fraction), round(height * fraction)
    boxes, scores = [], []
    for left, top in ((0, 0), (width - tw, 0), (0, height - th), (width - tw, height - th)):
        tile = rgb[top:top + th, left:left + tw]
        for detection in detector.detect(tile, confidence=.015):
            x1, y1, x2, y2 = detection.box
            if ((left and x1 <= .025 * tw) or (top and y1 <= .025 * th)
                    or (left + tw < width and x2 >= .975 * tw) or (top + th < height and y2 >= .975 * th)):
                continue
            boxes.append([x1 + left, y1 + top, x2 + left, y2 + top])
            scores.append(detection.confidence)
    return [{'box': boxes[i], 'score': scores[i]} for i in nms(boxes, scores, detector.nms_iou, max_detections=100)]


def evaluate(root, checkpoint, output, sizes, flip=False, nms_iou=.45, tiles=False):
    output.mkdir(parents=True, exist_ok=False)
    detector = NeuralDetector(checkpoint)
    suffix = 'tiles' if tiles else 'flip' if flip else ''
    runs = {str(size) + suffix: [] for size in sizes}
    times = defaultdict(list)
    for index, row in enumerate(validation_rows(root)):
        rgb = load_scene(root, row)
        h, w = rgb.shape[:2]
        truth = []
        for box in row['boxes']:
            coordinates = [v * s for v, s in zip(box['xyxy_normalized'], (w, h, w, h))]
            luma, texture = describe(rgb, coordinates)
            truth.append({'box': coordinates, 'category': box['category'], 'id': box['annotation_id'],
                          'luma': luma, 'background_texture': texture})
        for size in sizes:
            key = str(size) + suffix
            detector.size = size
            detector.nms_iou = nms_iou
            started = time.perf_counter()
            if tiles:
                predictions = tile_predictions(detector, rgb)
            else:
                detected = detector.detect(rgb[:, ::-1].copy() if flip else rgb, confidence=.001)
                boxes = [(w - d.box[2], d.box[1], w - d.box[0], d.box[3]) if flip else d.box for d in detected]
                predictions = [{'box': list(box), 'score': d.confidence} for box, d in zip(boxes, detected)]
            times[key].append(1000 * (time.perf_counter() - started))
            runs[key].append({'image_id': row['id'], 'truth': truth, 'size': [w, h],
                              'source': row.get('source_dataset', 'TACO'), 'predictions': predictions})
        if index % 40 == 0:
            print('scenes evaluated', index + 1, flush=True)
    summary = {name: operating_points(records) for name, records in runs.items()}
    if len(sizes) > 1:
        merged = []
        for views in zip(*runs.values()):
            combined = [p for view in views for p in view['predictions']]
            keep = nms([p['box'] for p in combined], [p['score'] for p in combined], .45, max_detections=100)
            merged.append(dict(views[0], predictions=[combined[i] for i in keep]))
        summary['multiscale'] = operating_points(merged)
        runs['multiscale'] = merged
    if '512' in runs:
        summary['miss_analysis'] = miss_analysis(runs['512'])
    summary['CPU_ms'] = {name: {'median': float(np.median(values[3:])), 'p95': float(np.percentile(values[3:], 95))}
                         for name, values in times.items()}
    summary['checkpoint_sha256'] = detector.checkpoint_sha256
    summary['nms_iou'] = nms_iou
    summary['test_opened'] = False
    (output / 'predictions.json').write_text(json.dumps(runs), encoding='utf-8')
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print({name: (summary[name]['mAP50'], summary[name]['mAP50_95'], summary[name]['points']['0.25']) for name in runs})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/detection/refine_D2'))
    parser.add_argument('--checkpoint', type=Path, default=Path('models/runtime/detector_D2.pth'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[512, 640, 768])
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--tiles', action='store_true')
    parser.add_argument('--nms-iou', type=float, default=.45)
    args = parser.parse_args()
    evaluate(args.root, args.checkpoint, args.output, args.sizes, args.flip, args.nms_iou, args.tiles)

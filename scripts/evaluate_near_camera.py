"""Measure material accuracy and whole-object localization in the close-object scenario."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image

from app.detector import NeuralDetector, crop_box, primary_detection
from app.localization import Region
from scripts.benchmark_material_transformer import classifier
from scripts.prepare_near_camera import digest, near_object
from scripts.train_scene_materials import scores
from training.detection_metrics import box_iou, match, spatial_metrics
from utils.config import CLASSES


def photos(path):
    with zipfile.ZipFile(path) as archive:
        for row in json.loads(archive.read('manifest.json')):
            if row.get('split', 'external') == 'train':
                continue
            path = row.get('path') or f"images/val/{row['id']:05d}.jpg"
            content = archive.read(path)
            if digest(content) != row['sha256']:
                raise ValueError(f"Photo changed: {row['id']}")
            with Image.open(BytesIO(content)) as image:
                rgb = np.asarray(image.convert('RGB'))
            w, h = row['size']
            if row.get('truth') is not None:
                truth = row['truth']
            else:
                truth = [b | {'box': [v * s for v, s in zip(b['xyxy_normalized'], (w, h, w, h))]}
                         for b in row['boxes']]
                truth = [b | {'near': near_object(b['box'], (w, h))} for b in truth]
            yield row['id'], rgb, truth


def coverage(detected, target):
    overlap = [max(0, min(detected[i+2], target[i+2]) - max(detected[i], target[i])) for i in range(2)]
    return float(np.prod(overlap) / ((target[2] - target[0]) * (target[3] - target[1])))


def evaluate_classification(archive, config, output):
    classify = classifier(config)
    by_type, confusion = {}, Counter()
    with zipfile.ZipFile(archive) as data:
        for row in json.loads(data.read('manifest.json')):
            content = data.read(row['path'])
            if digest(content) != row['sha256']:
                raise ValueError(f"Photo changed: {row['id']}")
            with Image.open(BytesIO(content)) as image:
                rgb = np.asarray(image.convert('RGB'))
            predicted = classify.classify(rgb).material
            summary = by_type.setdefault(row['category'], Counter())
            summary['objects'] += 1
            summary['correct'] += predicted == row['material']
            confusion[predicted] += 1
    totals = sum(by_type.values(), Counter())
    result = {'source': str(archive), 'scope': 'material classification on complete photos without annotated boxes',
              'objects': totals['objects'], 'correct': totals['correct'],
              'recall': totals['correct'] / totals['objects'], 'by_type': by_type, 'predicted_classes': confusion}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(output, totals, flush=True)


def evaluate(archive, config, output, camera_refinement=False):
    detector = NeuralDetector(config['detector_checkpoint'], config['cpu_threads'], config['detection_threshold'],
                              config['detection_nms_iou'], config.get('detection_views', ()))
    classify = classifier(config)
    detect = detector.detect_close if camera_refinement else detector.detect
    materials = {label: Counter() for label in CLASSES}
    clear_glass, clear_pet = Counter(), Counter()
    oracle_labels, oracle_estimates, detected_labels, detected_estimates = [], [], [], []
    false, foreground_predictions = Counter(), []
    detection_times, material_times, end_times = [], [], []
    records = []
    for index, (ident, rgb, truth) in enumerate(photos(archive)):
        if len(truth) != 1 or not truth[0]['material'] or not truth[0]['near']:
            continue
        target = truth[0]
        label = target['material']
        area = target['box']
        by_class = materials[label]
        by_class['objects'] += 1
        h, w = rgb.shape[:2]
        region = Region(crop_box(area, w, h), 'annotated-waste')
        oracle = classify.classify(rgb, region=region).material
        oracle_labels.append(CLASSES.index(label))
        oracle_estimates.append(CLASSES.index(oracle))
        by_class['classified_on_annotation'] += oracle == label
        started = time.perf_counter()
        detections = detect(rgb, confidence=.001)
        detection_ms = (time.perf_counter() - started) * 1000
        rankings = [{'box': item.box, 'score': item.confidence} for item in detections]
        active = [item for item in detections if item.confidence >= config['detection_threshold']]
        primary = primary_detection(active)
        started_material = time.perf_counter()
        reading = classify.classify(rgb, region=primary.region) if primary else None
        if primary:
            material_times.append((time.perf_counter() - started_material) * 1000)
        detection_times.append(detection_ms)
        end_times.append(detection_ms + (time.perf_counter() - started_material) * 1000)
        maximum_iou = max((box_iou(item.box, area) for item in detections), default=0.)
        active_matches = match([{'box': item.box, 'score': item.confidence} for item in active], truth)
        by_class['any_localized'] += bool(active_matches)
        false['other_boxes_in_single_object_frames'] += len(active) - len(active_matches)
        located = primary is not None and box_iou(primary.box, area) >= .50
        whole = located and coverage(primary.box, area) >= .85
        by_class['primary_localized'] += located
        by_class['primary_coverage85'] += whole
        by_class['joint_correct'] += located and reading.material == label
        by_class['joint_complete'] += whole and reading.material == label
        if located:
            detected_labels.append(CLASSES.index(label))
            detected_estimates.append(CLASSES.index(reading.material))
        if not located:
            miss = ('another_box_won' if active_matches else 'below_threshold' if maximum_iou >= .50 else
                    'partial_or_imprecise_box' if maximum_iou >= .10 else 'no_overlapping_box')
            by_class[miss] += 1
        if target.get('source_material') == 'Glass - Non-coloured' or target['category'] == 'Clear Glass (GL)':
            clear_glass.update({'objects': 1, 'oracle_correct': int(oracle == label), 'primary_localized': int(located),
                                'joint_correct': int(located and reading.material == label)})
        if target['category'] == 'Clear PET':
            clear_pet.update({'objects': 1, 'primary_localized': int(located),
                              'joint_correct': int(located and reading.material == label)})
        foreground_predictions.append({'image_id': ident, 'truth': [{'box': area, 'category': 'residuo'}],
                                       'predictions': rankings, 'size': [w, h]})
        records.append({'id': ident, 'class': label, 'oracle': oracle, 'primary_answer': reading.material if reading else None,
                        'primary_box': primary.box if primary else None,
                        'primary_iou': box_iou(primary.box, area) if primary else 0,
                        'primary_coverage': coverage(primary.box, area) if primary else 0,
                        'maximum_returned_IoU': maximum_iou})
        if len(records) % 80 == 0:
            print('close scenes evaluated', len(records), 'frames read', index + 1, flush=True)
    if not records:
        raise ValueError('No frames contain a single nearby object with one of the known materials.')
    detection = spatial_metrics(foreground_predictions, config['detection_threshold'])
    totals = sum(materials.values(), Counter())
    latency = {key: {'median': float(np.median(times[3:])), 'p95': float(np.percentile(times[3:], 95))}
               for key, times in (('detector_AP_scoring', detection_times), ('primary_material', material_times),
                                  ('combined_AP_scoring', end_times)) if len(times) > 3}
    result = {'source': str(archive), 'independent_single_near_frames': len(records),
              'camera_refinement': camera_refinement,
              'detector_checkpoint_sha256': detector.checkpoint_sha256,
              'classifiers': {
                  'base': classify.checkpoint_sha256, 'candidate': classify.candidate_checkpoint_sha256,
                  'refiner': classify.refiner_checkpoint_sha256},
              'primary_near_recall': totals['primary_localized'] / len(records),
              'primary_near_joint_recall': totals['joint_correct'] / len(records),
              'primary_full_coverage_recall': totals['primary_coverage85'] / len(records),
              'primary_near_whole_object_joint_recall': totals['joint_complete'] / len(records),
              'classification_on_annotation': scores(oracle_labels, oracle_estimates),
              'classification_given_primary_match': scores(detected_labels, detected_estimates),
              'detection': {key: detection[key] for key in ('precision', 'recall', 'mAP50', 'mAP50_95')},
              'per_material': materials, 'clear_glass': clear_glass, 'clear_PET': clear_pet,
              'false_positives': false, 'CPU_ms': latency, 'frames': records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(output, 'frames', len(records), 'joint', totals['joint_correct'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, default=Path('config/app.json'))
    parser.add_argument('--classification-only', action='store_true')
    parser.add_argument('--camera-refinement', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.classification_only:
        evaluate_classification(args.archive, config, args.output)
    else:
        evaluate(args.archive, config, args.output, camera_refinement=args.camera_refinement)

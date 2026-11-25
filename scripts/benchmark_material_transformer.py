"""Measure Vision-IA with identical D3 detections and two material policies."""
import argparse
import json
from pathlib import Path
import statistics
import time

from app.detector import NeuralDetector, primary_detection
from app.vision import InferenceService
from scripts.diagnose_scene_detector import load_scene, validation_rows
from scripts.evaluate_detector import MATERIALS
from training.detection_metrics import match


def classifier(config):
    return InferenceService(config['checkpoint'], config['cpu_threads'], config.get('material_candidate_checkpoint'),
                            config.get('material_candidate_threshold', .4), config.get('material_candidate_weight'),
                            config.get('material_refiner_checkpoint'), config.get('material_refiner_weight'),
                            config.get('material_refiner_preserve', ()), config.get('material_refiner_calibration'),
                            config.get('material_refiner_preserve_consensus', ()))


def benchmark(baseline, candidate):
    keys = ('detector_checkpoint', 'detection_threshold', 'detection_nms_iou', 'detection_views', 'cpu_threads')
    if any(baseline[key] != candidate[key] for key in keys):
        raise ValueError('Both systems must use identical localization and CPU threads.')
    detector = NeuralDetector(baseline['detector_checkpoint'], baseline['cpu_threads'], baseline['detection_threshold'],
                              baseline['detection_nms_iou'], baseline['detection_views'])
    engines = [classifier(config) for config in (baseline, candidate)]
    frame_times, material_times = [[], []], [[], []]
    correct, primary_correct = [0, 0], [0, 0]
    total = 0
    root = Path('data/detection/refine_D2')
    for index, row in enumerate(validation_rows(root)):
        rgb = load_scene(root, row)
        h, w = rgb.shape[:2]
        truth = [{'box': [v * s for v, s in zip(box['xyxy_normalized'], (w, h, w, h))],
                  'material': MATERIALS.get(box['category'])} for box in row['boxes']]
        total += sum(t['material'] is not None for t in truth)
        start = time.perf_counter()
        detections = detector.detect(rgb)
        primary = primary_detection(detections)
        detector_ms = 1000 * (time.perf_counter() - start)
        pairs = match([{'box': detection.box, 'score': detection.confidence} for detection in detections], truth)
        for side in (index % 2, (index + 1) % 2):
            start = time.perf_counter()
            reading = engines[side].classify(rgb, region=primary.region) if primary else None
            material_ms = 1000 * (time.perf_counter() - start)
            if index >= 5:
                frame_times[side].append(detector_ms + material_ms)
                if primary:
                    material_times[side].append(material_ms)
            for i, j, _ in pairs:
                expected = truth[j]['material']
                if expected is None:
                    continue
                is_primary = detections[i] is primary
                result = reading if is_primary else engines[side].classify(rgb, region=detections[i].region)
                correct[side] += result.material == expected
                primary_correct[side] += is_primary and result.material == expected

    def stats(values):
        return {'median': statistics.median(values), 'p95': statistics.quantiles(values, n=100)[94], 'count': len(values)}

    report = {'detector_sha256': detector.checkpoint_sha256, 'test_used': False, 'cpu_threads': baseline['cpu_threads']}
    for i, name in enumerate(('baseline', 'candidate')):
        report[name] = {'joint_correct': correct[i], 'joint_primary_correct': primary_correct[i], 'joint_objects': total,
                        'joint_recall': correct[i] / total, 'joint_primary_recall': primary_correct[i] / total,
                        'CPU_pipeline_ms': stats(frame_times[i]), 'CPU_material_ms': stats(material_times[i])}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('baseline', 'candidate', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    configurations = [json.loads(p.read_text(encoding='utf-8')) for p in (args.baseline, args.candidate)]
    args.output.write_text(json.dumps(benchmark(*configurations), indent=2), encoding='utf-8')

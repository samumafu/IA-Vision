"""Add D3 crops matched to human material annotations from TACO train only."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from io import BytesIO
from pathlib import Path
import zipfile

from PIL import Image

from app.detector import NeuralDetector
from app.vision import InferenceService
from scripts.diagnose_scene_detector import load_scene, validation_rows
from scripts.evaluate_scene_materials import predict
from scripts.prepare_scene_materials import encode
from training.detection_metrics import match
from utils.config import CLASSES


def build(config_path, root, scenes, output):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    detector = NeuralDetector(config['detector_checkpoint'], confidence=config['detection_threshold'],
                              nms_iou=config['detection_nms_iou'], views=config['detection_views'])
    validation_rows(root)  # Audit split/group hashes without opening validation or test photos.
    source_rows = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    train_photos = {r['id']: r for r in source_rows if r['split'] == 'train' and r.get('source_dataset', 'TACO') == 'TACO'}
    with zipfile.ZipFile(scenes) as archive:
        scene_rows = json.loads(archive.read('manifest.json'))
    objects = defaultdict(dict)
    for row in scene_rows:
        if row['split'] == 'train' and row.get('dataset', 'TACO') == 'TACO':
            photo = train_photos[row['image_id']]
            if row['source_sha256'] != photo['sha256'] or row['group'] != photo['group']:
                raise ValueError('Scene annotations do not match the protected train manifest.')
            objects[row['image_id']][row['id']] = row
    counts, rows = Counter(), []
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_STORED) as archive:
        for index, (image_id, labels) in enumerate(objects.items(), 1):
            photo = train_photos[image_id]
            rgb = load_scene(root, photo)
            h, w = rgb.shape[:2]
            truth = [{'id': box['annotation_id'], 'box': [v * s for v, s in zip(box['xyxy_normalized'], (w, h, w, h))]}
                     for box in photo['boxes']]
            detections = detector.detect(rgb)
            proposals = [{'box': d.box, 'score': d.confidence} for d in detections]
            for i, j, iou in match(proposals, truth, threshold=.53):
                label = labels.get(truth[j]['id'])
                if label is None:
                    continue
                x1, y1, x2, y2 = detections[i].region.box
                content = encode(rgb[y1:y2, x1:x2])
                path = f'train/{image_id}_{label["id"]}.jpg'
                archive.writestr(path, content)
                rows.append({key: label[key] for key in ('id', 'image_id', 'source_sha256', 'class', 'split')} |
                            {'path': path, 'sha256': hashlib.sha256(content).hexdigest(), 'IoU': iou})
                counts[label['class']] += 1
            if index % 100 == 0:
                print(f'{index}/{len(objects)} TACO train photos — {len(rows)} crops', flush=True)
        audit = {'split': 'train', 'test_images_opened': 0, 'detector_sha256': detector.checkpoint_sha256,
                 'minimum_IoU': .53, 'photos': len(objects), 'crops_by_material': counts}
        archive.writestr('manifest.json', json.dumps(rows))
        archive.writestr('audit.json', json.dumps(audit, indent=2))
    temporary.replace(output)
    print(json.dumps(audit, indent=2), flush=True)


def rank_hard(config_path, source, output):
    """Give additional sampling weight to hard human-labelled training objects."""
    config = json.loads(config_path.read_text(encoding='utf-8'))
    models = {key: InferenceService(config[path]) for key, path in
              (('refined', 'checkpoint'), ('scene', 'material_candidate_checkpoint'))}
    weight = config['material_candidate_weight']
    with zipfile.ZipFile(source) as data:
        rows = json.loads(data.read('manifest.json'))
        audit = json.loads(data.read('audit.json'))
        if audit['split'] != 'train' or audit['test_images_opened'] != 0 or any(r['split'] != 'train' for r in rows):
            raise ValueError('Difficulty can be ranked only on training crops.')
        pictures = []
        for row in rows:
            content = data.read(row['path'])
            if hashlib.sha256(content).hexdigest() != row['sha256']:
                raise ValueError('Detector crop was modified.')
            with Image.open(BytesIO(content)) as picture:
                pictures.append(picture.convert('RGB'))
        probabilities = predict(models, pictures)
        fused = (1 - weight) * probabilities['refined'] + weight * probabilities['scene']
        wanted = {'plastico', 'papel', 'carton', 'organicos'}
        hard = Counter()
        for row, predicted in zip(rows, fused.argmax(1), strict=True):
            expected, classified = row['class'], CLASSES[predicted]
            row['sampling_weight'] = 1.8 if classified != expected and {classified, expected} <= wanted else 1.
            if row['sampling_weight'] > 1:
                hard[expected] += 1
        audit['prior_cnn_sha256'] = {key: model.checkpoint_sha256 for key, model in models.items()}
        audit['hard_by_material'] = hard
        with zipfile.ZipFile(output, 'x', zipfile.ZIP_STORED) as archive:
            for row in rows:
                archive.writestr(row['path'], data.read(row['path']))
            archive.writestr('manifest.json', json.dumps(rows))
            archive.writestr('audit.json', json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/app.json', type=Path)
    parser.add_argument('--root', default='data/detection/refine_D2', type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--scenes', type=Path)
    source.add_argument('--rank-hard-from', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    opts = parser.parse_args()
    if opts.rank_hard_from:
        rank_hard(opts.config, opts.rank_hard_from, opts.output)
    else:
        build(opts.config, opts.root, opts.scenes, opts.output)

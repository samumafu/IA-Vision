"""Compare runtime classifiers on the frozen external crops and detected boxes."""
import argparse
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image

from app.detector import crop_box
from app.localization import Region
from scripts.benchmark_material_transformer import classifier
from scripts.evaluate_near_camera import photos
from scripts.train_scene_materials import digest, scores
from utils.config import CLASSES


def compare(configs, output, folder):
    models = {key: classifier(config) for key, config in configs.items()}
    reference = json.loads((folder / 'bower-camera-integrated.json').read_text(encoding='utf-8'))
    rows = {str(row['id']): row for row in reference['frames']}
    truths, on_annotation = [], defaultdict(list)
    localized_truth, localized_material = [], defaultdict(list)
    joint = defaultdict(Counter)
    paired = defaultdict(Counter)
    groups = defaultdict(lambda: defaultdict(Counter))
    durations = defaultdict(list)
    evaluated = set()
    for ident, rgb, objects in photos(folder / 'bower.zip'):
        row = rows.get(str(ident))
        if row is None:
            continue
        evaluated.add(str(ident))
        target = objects[0]
        expected = target['material']
        if expected != row['class'] or not target['near'] or len(objects) != 1:
            raise ValueError(f'Frozen Bower protocol changed: {ident}')
        h, w = rgb.shape[:2]
        annotated = Region(crop_box(target['box'], w, h), 'manual')
        category = CLASSES.index(expected)
        truths.append(category)
        matched = row['primary_iou'] >= .5
        detected = Region(crop_box(row['primary_box'], w, h), 'fixed-D3') if matched else None
        if matched:
            localized_truth.append(category)
        group = f"{target['source_material']}/{target['category']}"
        order = list(models)
        if len(evaluated) % 2:
            order.reverse()
        for key in order:
            engine = models[key]
            observed = engine.classify(rgb, region=annotated).material
            on_annotation[key].append(CLASSES.index(observed))
            if observed == expected and row['oracle'] != expected:
                paired[key]['annotation_gained'] += 1
            elif observed != expected and row['oracle'] == expected:
                paired[key]['annotation_lost'] += 1
            counts = groups[key][group]
            counts['support'] += 1
            counts[f'predicted_{observed}'] += 1
            if matched:
                start = time.perf_counter()
                observed = engine.classify(rgb, region=detected).material
                durations[key].append((time.perf_counter() - start) * 1000)
                localized_material[key].append(CLASSES.index(observed))
                joint[key][expected] += observed == expected
                if observed == expected and row['primary_answer'] != expected:
                    paired[key]['joint_gained'] += 1
                elif observed != expected and row['primary_answer'] == expected:
                    paired[key]['joint_lost'] += 1
    if evaluated != set(rows):
        raise ValueError('Frozen set of external frames changed.')
    organics = defaultdict(Counter)
    with zipfile.ZipFile(folder / 'bdwaste-organics.zip') as archive:
        for row in json.loads(archive.read('manifest.json')):
            content = archive.read(row['path'])
            if digest(content) != row['sha256']:
                raise ValueError(f"Changed organic evaluation photo: {row['id']}")
            with Image.open(BytesIO(content)) as image:
                rgb = np.asarray(image.convert('RGB'))
            for name, engine in models.items():
                observed = engine.classify(rgb).material
                organics[name]['objects'] += 1
                organics[name][f'predicted_{observed}'] += 1
    results = {}
    for key in models:
        results[key] = {'Bower_on_annotation': scores(truths, on_annotation[key]),
                        'Bower_on_matched_detection': scores(localized_truth, localized_material[key]),
                        'Bower_joint_recall': sum(joint[key].values()) / len(rows),
                        'Bower_joint_correct_by_class': joint[key],
                        'Bower_paired_changes_vs_reference': paired[key],
                        'BDWaste_organic_photos': organics[key],
                        'Bower_recall_per_source_label': groups[key],
                        'CPU_classifier_ms': {'median': float(np.median(durations[key][4:])),
                                              'p95': float(np.percentile(durations[key][4:], 95))}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    for key, report in results.items():
        print(key, report['Bower_on_annotation']['accuracy'], report['Bower_on_annotation']['macro_F1_present'],
              report['Bower_joint_recall'], report['BDWaste_organic_photos'].get('predicted_organicos', 0), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--configs', required=True, nargs='+')
    parser.add_argument('--folder', type=Path, default=Path('runs/near_camera'))
    args = parser.parse_args()
    configs = dict(arg.split('=', 1) for arg in args.configs)
    if len(configs) != len(args.configs):
        parser.error('Distinct names are required for each configuration.')
    compare({name: json.loads(Path(path).read_text(encoding='utf-8')) for name, path in configs.items()},
            args.output, args.folder)

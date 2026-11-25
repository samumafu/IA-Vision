"""Add usable SWiM3 plastic and metal box crops to TACO scene training."""

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from app.detector import crop_box
from scripts.prepare_scene_materials import encode, rectangular_box
from scripts.prepare_mwcd import protected_photos
from utils.visual_audit import perceptual


def usable_boxes(row, width, height):
    boxes = np.asarray([r['xyxy_normalized'] for r in row['boxes']]) * [width, height, width, height]
    labels = [r['category'] for r in row['boxes']]
    counts = Counter()
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    for index in np.argsort(-areas):
        label, bounds = labels[index], boxes[index]
        if label not in ('plastico', 'metal') or counts[label] >= 2 or min(bounds[2:] - bounds[:2]) < 24:
            continue
        overlap = np.maximum(0., np.minimum(bounds[2:], boxes[:, 2:]) - np.maximum(bounds[:2], boxes[:, :2])).prod(axis=1)
        if any(a >= .20 * areas[index] and other != label for a, other in zip(overlap, labels)):
            continue
        counts[label] += 1
        yield index, label, bounds


def build(scenes, swim, output):
    with zipfile.ZipFile(scenes) as original, zipfile.ZipFile(swim) as additional:
        audit = json.loads(original.read('audit.json'))
        incoming_audit = json.loads(additional.read('audit.json'))
        if incoming_audit.get('contains_only_added_images') is not True or incoming_audit['test_images_opened'] != 0:
            raise ValueError('SWiM3 input must be the audited training-only supplement.')
        rows = json.loads(original.read('manifest.json'))
        incoming_rows = json.loads(additional.read('manifest.json'))
        counts = Counter()
        images = Counter()
        sources = set()
        excluded = 0
        exact, protected, _ = protected_photos()
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, 'x', zipfile.ZIP_STORED) as archive:
            for name in original.namelist():
                if name not in ('manifest.json', 'audit.json'):
                    archive.writestr(name, original.read(name))
            for row in incoming_rows:
                if row.get('source_dataset') != 'SWiM3':
                    continue
                if row['split'] != 'train':
                    raise ValueError('SWiM3 scene photos must all be in train.')
                ident = row['id']
                content = additional.read(f'images/train/{ident:05d}.jpg')
                if hashlib.sha256(content).hexdigest() != row['sha256']:
                    raise ValueError(f'SWiM3 image changed: {ident}')
                with Image.open(BytesIO(content)) as picture:
                    rgb = np.asarray(picture.convert('RGB'))
                h, w = rgb.shape[:2]
                used = set()
                for index, label, bounds in usable_boxes(row, w, h):
                    views = {}
                    for view, coords in (('square', crop_box(bounds, w, h)), ('tight', rectangular_box(bounds, w, h, 0)),
                                         ('rect', rectangular_box(bounds, w, h, .12))):
                        x1, y1, x2, y2 = coords
                        views[view] = encode(rgb[y1:y2, x1:x2])
                    if any(hashlib.sha256(content).hexdigest() in exact for content in views.values()):
                        excluded += 1
                        continue
                    with Image.open(BytesIO(views['square'])) as picture:
                        if protected.matches(*perceptual(picture)):
                            excluded += 1
                            continue
                    paths = {}
                    for view, content in views.items():
                        name = f'train/{label}/SWiM3_{ident}_{index}_{view}.jpg'
                        archive.writestr(name, content)
                        paths[view] = name
                    rows.append({'id': f'SWiM3_{ident}_{index}', 'image_id': ident, 'group': row['source_git_blob'],
                                 'split': 'train', 'class': label, 'category': f'SWiM3 {label}', 'paths': paths,
                                 'dataset': 'SWiM3', 'source_sha256': row['source_photo_sha256']})
                    counts[label] += 1
                    used.add(label)
                if used:
                    images.update(used)
                    sources.add(ident)
            audit['SWiM3'] = {'crops': counts, 'photos_by_material': images, 'distinct_photos': len(sources),
                             'source': incoming_audit['source'], 'source_split': 'train', 'minimum_side_pixels': 24,
                             'excluded_protected_crops': excluded}
            archive.writestr('manifest.json', json.dumps(rows))
            archive.writestr('audit.json', json.dumps(audit, indent=2))
    print(json.dumps(audit['SWiM3'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scenes', 'swim', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    build(args.scenes, args.swim, args.output)

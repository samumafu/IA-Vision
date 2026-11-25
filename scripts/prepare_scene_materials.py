"""Build an isolated TACO material dataset from verified photos and human polygons."""
import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image

from app.detector import crop_box
from scripts.diagnose_scene_detector import load_scene, validation_rows


MATERIALS = {
    'plastico': ('Clear plastic bottle', 'Other plastic bottle', 'Plastic bottle cap', 'Disposable plastic cup',
                 'Other plastic cup', 'Plastic lid', 'Other plastic', 'Plastic film', 'Six pack rings',
                 'Garbage bag', 'Single-use carrier bag', 'Polypropylene bag', 'Plastic glooves',
                 'Plastic utensils', 'Plastic straw', 'Other plastic container', 'Styrofoam piece', 'Tupperware'),
    'metal': ('Aluminium foil', 'Metal bottle cap', 'Food Can', 'Aerosol', 'Drink can', 'Metal lid', 'Pop tab', 'Scrap metal'),
    'vidrio': ('Glass bottle', 'Broken glass', 'Glass cup', 'Glass jar'),
    'carton': ('Toilet tube', 'Egg carton', 'Corrugated carton', 'Pizza box'),
    'papel': ('Magazine paper', 'Tissues', 'Normal paper', 'Paper bag'),
    'organicos': ('Food waste',),
}
LABELS = {name: label for label, names in MATERIALS.items() for name in names}


def rectangular_box(box, width, height, margin):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, int(np.floor(x1 - dx))), max(0, int(np.floor(y1 - dy))),
            min(width, int(np.ceil(x2 + dx))), min(height, int(np.ceil(y2 + dy))))


def object_mask(annotation, region, scale):
    left, top, right, bottom = region
    mask = np.zeros((bottom - top, right - left), np.uint8)
    polygons = annotation['segmentation']
    if not isinstance(polygons, list):
        return mask
    for segment in polygons:
        if len(segment) >= 6:
            points = np.asarray(segment).reshape(-1, 2) * scale - (left, top)
            cv2.fillPoly(mask, [points.round().astype(np.int32)], 1)
    return mask


def encode(image):
    image = Image.fromarray(image)
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    output = BytesIO()
    image.save(output, format='JPEG', quality=95)
    return output.getvalue()


def build(root, annotations, protected, output):
    validation_rows(root)
    manifest = root / 'manifest.json'
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    original = json.loads(annotations.read_text(encoding='utf-8'))
    objects = {a['id']: a for a in original['annotations']}
    shapes = {i['id']: (i['width'], i['height']) for i in original['images']}
    protected = json.loads(protected.read_text(encoding='utf-8'))
    exact, pixels = set(protected['sha256']), set(protected['pixels_sha256'])
    records, excluded = [], defaultdict(Counter)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = {}
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_STORED) as archive:
        for i, row in enumerate(rows):
            split = row['split']
            if split not in ('train', 'val') or row.get('source_dataset') == 'MJU-Waste':
                continue
            if {row['sha256'], row.get('downloaded_sha256')} & exact or row['pixels_sha256'] in pixels:
                excluded[split]['photo_matches_B'] += 1
                continue
            if not any(b['category'] in LABELS for b in row['boxes']):
                continue
            rgb = load_scene(root, row)
            h, w = rgb.shape[:2]
            actual = {b['annotation_id']: np.asarray(b['xyxy_normalized']) * [w, h, w, h] for b in row['boxes']}
            for box in row['boxes']:
                ident, category = box['annotation_id'], box['category']
                if category not in LABELS:
                    excluded[split]['ambiguous_category'] += 1
                    continue
                label = LABELS[category]
                coordinates = actual[ident]
                width, height = coordinates[2:] - coordinates[:2]
                if min(width, height) < 14:
                    excluded[split]['small_or_thin'] += 1
                    continue
                scale = (w / shapes[row['id']][0], h / shapes[row['id']][1])
                annotation = objects[ident]
                if annotation['image_id'] != row['id'] or annotation.get('iscrowd'):
                    raise ValueError(f'Incompatible source annotation {ident}')
                tight = rectangular_box(coordinates, w, h, 0)
                mask = object_mask(annotation, tight, scale)
                if mask.mean() < .15:
                    excluded[split]['sparse_or_invalid_polygon'] += 1
                    continue
                contaminated = False
                for other in row['boxes']:
                    if other['annotation_id'] == ident or LABELS.get(other['category']) == label:
                        continue
                    bounds = actual[other['annotation_id']]
                    intersection = np.maximum(0, np.minimum(coordinates[2:], bounds[2:]) - np.maximum(coordinates[:2], bounds[:2]))
                    if intersection.prod() / (width * height) >= .22:
                        contaminated = True
                        break
                if contaminated:
                    excluded[split]['overlap_other_material'] += 1
                    continue
                square = crop_box(coordinates, w, h)
                rect = rectangular_box(coordinates, w, h, .12)
                left, top, right, bottom = tight
                object_rgb = rgb[top:bottom, left:right]
                variants = {'tight': object_rgb, 'square': rgb[square[1]:square[3], square[0]:square[2]],
                            'rect': rgb[rect[1]:rect[3], rect[0]:rect[2]],
                            'foreground': np.where(mask[..., None] > 0, object_rgb, 128).astype(np.uint8)}
                if split == 'val':
                    variants['background'] = np.where(mask[..., None] > 0, 128, object_rgb).astype(np.uint8)
                encoded = {name: encode(view) for name, view in variants.items()}
                content_hashes = {hashlib.sha256(content).hexdigest() for content in encoded.values()}
                if content_hashes & exact or any(seen.get(key, split) != split for key in content_hashes):
                    excluded[split]['crop_matches_other_split_or_B'] += 1
                    continue
                seen.update(dict.fromkeys(content_hashes, split))
                paths = {}
                for name, content in encoded.items():
                    path = f'{split}/{label}/{row["id"]}_{ident}_{name}.jpg'
                    archive.writestr(path, content)
                    paths[name] = path
                records.append({'id': ident, 'image_id': row['id'], 'group': row['group'], 'split': split, 'class': label,
                                'category': category, 'paths': paths, 'size': [width, height], 'coverage': float(mask.mean()),
                                'source_sha256': row['sha256']})
            if i % 200 == 0:
                print('photos checked', i, flush=True)
        audit = {'counts': {s: Counter(r['class'] for r in records if r['split'] == s) for s in ('train', 'val')},
                 'excluded': excluded, 'mapping': {c['name']: LABELS.get(c['name']) for c in original['categories']},
                 'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 'annotation_sha256': hashlib.sha256(annotations.read_bytes()).hexdigest(), 'test_images_opened': 0}
        archive.writestr('manifest.json', json.dumps(records))
        archive.writestr('audit.json', json.dumps(audit, indent=2))
    print(json.dumps(audit['counts']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(Path('data/detection/refine_D2'), Path('data/detection/taco/annotations.json'),
          Path('exports/protected_classification_hashes.json'), args.output)

"""Package independently annotated close-object data without opening reserved tests."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import zipfile

import imagehash
import numpy as np
from PIL import Image, ImageOps

from app.detector import crop_box
from scripts.prepare_mwcd import NearPhotos, protected_photos
from scripts.prepare_scene_materials import encode, rectangular_box
from utils.visual_audit import perceptual


MMORWAD = 'https://doi.org/10.5281/zenodo.22226868'
BOWER = 'https://huggingface.co/datasets/BowerApp/bower-waste-annotations'
BDWASTE = 'https://doi.org/10.17632/96g5pgfnfw.1'
MMORWAD_MD5 = {
    'color_img.zip': '0dd9d57cbcdd9e1fd6a5b43f5d7b8c00',
    'dataset.csv': 'dcfa53d6d62aa00c054805576fb4ecf1',
    'coco_annotations.json': '565caceddbddc6af395f3cca3053671a',
    'sample_splits.csv': '5a170ce7ea0abdb80e734fe360bd9323',
}
BOWER_MATERIALS = {
    'Plastic - Hard': 'plastico', 'Plastic - Soft': 'plastico', 'Metal': 'metal',
    'Glass - Non-coloured': 'vidrio', 'Glass - Coloured': 'vidrio', 'Fiber - Paper': 'papel',
    'Fiber - Cardboard / Paper Packaging': 'carton',
}
MMORWAD_MATERIALS = {
    'Clear Glass (GL)': 'vidrio', 'Brown Glass (GL)': 'vidrio', 'Green Glass (GL)': 'vidrio',
    'Aluminium (ALU)': 'metal', 'Steel (FE)': 'metal', 'Paper (PAP)': 'papel',
    'Cardboard (PAP)': 'carton', 'Paperboard (PAP)': 'carton',
    **dict.fromkeys((
        'Polyethylene Terephthalate (PET)', 'Polypropylene (PP)', 'High Density Polyethylene (HDPE)',
        'Other Plastic (Other)', 'Low Density Polyethylene (LDPE)', 'Polystyrene (PS)',
        'Polyvinyl Chloride (PVC)',
    ), 'plastico'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def new_photo_filter():
    exact, classification, detection = protected_photos()
    for path in ('runs/intensive/scene-materials-swim3.zip', 'runs/intensive/mwcd-train.zip'):
        with zipfile.ZipFile(path) as archive:
            for row in json.loads(archive.read('manifest.json')):
                exact.add(row['source_sha256'])
                if row.get('sha256'):
                    exact.add(row['sha256'])
    for path in Path('data/field_samples/samples').glob('*/original.png'):
        exact.add(digest(path.read_bytes()))
        with Image.open(path) as image:
            p, d = perceptual(image)
            classification.add(p, d, 'field')
    return exact, classification, detection


def is_known(image, exact, classification, detection, content):
    if digest(content) in exact:
        return True
    for view in (image, ImageOps.mirror(image)):
        p, d = perceptual(view)
        if classification.matches(p, d) or detection.matches(int(str(imagehash.phash(view)), 16)):
            return True
    return False


def jpg(image):
    data = BytesIO()
    image.save(data, format='JPEG', quality=94, subsampling=0)
    return data.getvalue()


def near_object(box, size):
    w, h = size
    return ((box[2] - box[0]) * (box[3] - box[1])) ** .5 * 512 / max(w, h) >= 64


def bower(source, output):
    import pyarrow.parquet as pq

    exact, classification, detection = new_photo_filter()
    duplicates, rows, excluded = NearPhotos(), {}, Counter()
    parquet = pq.ParquetFile(source)
    # Group every box for an image before deciding whether it has one object.
    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            ident = row['image_id']
            size = (row['image_width'], row['image_height'])
            content = row['image']['bytes']
            image = rows.setdefault(ident, {'size': size, 'content': content, 'boxes': []})
            if image['size'] != size or image['content'] != content:
                raise ValueError(f'Conflicting photo identity {ident}')
            x1, y1, x2, y2 = (row[key] for key in ('x1', 'y1', 'x2', 'y2'))
            w, h = size
            box = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
            if min(box[2] - box[0], box[3] - box[1]) < 2:
                excluded['invalid_box'] += 1
                continue
            material, category = row['material'], row['object']
            label = BOWER_MATERIALS.get(material)
            if label == 'carton' and category != 'Box/Card/Tray/Crate':
                label = None
            image['boxes'].append({'box': box, 'material': label, 'source_material': material,
                                   'category': category, 'near': near_object(box, size)})
    records = []
    exact_seen = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output.with_suffix('.tmp'), 'x') as archive:
        for ident, row in sorted(rows.items()):
            if not any(box['near'] and box['material'] for box in row['boxes']):
                excluded['no_near_mapped_object'] += 1
                continue
            with Image.open(BytesIO(row['content'])) as raw:
                image = raw.convert('RGB')
                if image.size != row['size']:
                    image = ImageOps.exif_transpose(raw).convert('RGB')
                if image.size != row['size']:
                    excluded['dimensions_mismatch'] += 1
                    continue
            p, d = perceptual(image)
            pm, dm = perceptual(ImageOps.mirror(image))
            source_hash = digest(row['content'])
            if (is_known(image, exact, classification, detection, row['content']) or
                    source_hash in exact_seen or duplicates.matches(p, d) or duplicates.matches(pm, dm)):
                excluded['exact_or_near_duplicate'] += 1
                continue
            duplicates.add(p, d, ident)
            exact_seen.add(source_hash)
            path = f'images/{ident}.jpg'
            data = jpg(image)
            archive.writestr(path, data)
            records.append({'id': ident, 'size': row['size'], 'truth': row['boxes'], 'path': path,
                            'source_sha256': source_hash, 'sha256': digest(data)})
        audit = {'source': BOWER, 'role': 'external evaluation only', 'license': 'MIT',
                 'source_images': len(rows), 'source_objects': parquet.metadata.num_rows,
                 'included_images': len(records), 'excluded': excluded,
                 'near_objects': Counter(box['material'] for r in records for box in r['truth']
                                         if box['near'] and box['material']),
                 'single_near_objects': Counter(r['truth'][0]['material'] for r in records if len(r['truth']) == 1)}
        archive.writestr('manifest.json', json.dumps(records))
        archive.writestr('audit.json', json.dumps(audit, indent=2))
    output.with_suffix('.tmp').replace(output)
    print(json.dumps(audit, indent=2), flush=True)


def mmorwad(source, output, image_output):
    for name, reference in MMORWAD_MD5.items():
        with (source / name).open('rb') as stream:
            if hashlib.file_digest(stream, 'md5').hexdigest() != reference:
                raise ValueError(f'Published checksum mismatch: {name}')
    with (source / 'dataset.csv').open(encoding='utf-8-sig', newline='') as stream:
        metadata = {int(r['ID']): r for r in csv.DictReader(stream)}
    with (source / 'sample_splits.csv').open(encoding='utf-8-sig', newline='') as stream:
        partitions = {int(r['ID']): r for r in csv.DictReader(stream)}
    groups = defaultdict(set)
    for row in partitions.values():
        groups[row['item_name'].strip().casefold()].add(row['split'])
    if any(len(values) != 1 for values in groups.values()):
        raise ValueError('The published product groups overlap.')
    coco = read_json(source / 'coco_annotations.json')
    boxes = defaultdict(list)
    for row in coco['annotations']:
        boxes[row['image_id']].append(row)
    exact, classification, detection = new_photo_filter()
    # Exclude external evaluation matches from development before using either set for training.
    external = output.parent / 'bower.zip'
    if not external.is_file():
        raise FileNotFoundError('Build the external evaluation package before packaging training.')
    with zipfile.ZipFile(external) as archive:
        for row in json.loads(archive.read('manifest.json')):
            exact.add(row['source_sha256'])
            with Image.open(BytesIO(archive.read(row['path']))) as image:
                p, d = perceptual(image)
                classification.add(p, d, 'external')
    excluded, manifests, detector_rows = Counter(), [], []
    duplicates, source_hashes = NearPhotos(), set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (zipfile.ZipFile(source / 'color_img.zip') as incoming,
          zipfile.ZipFile(output.with_suffix('.tmp'), 'x') as materials,
          zipfile.ZipFile(image_output.with_suffix('.tmp'), 'x') as detector):
        names = {Path(member).name: member for member in incoming.namelist()}
        # Keep validation independent, even if a near duplicate spans two product labels.
        for row in sorted(coco['images'], key=lambda r: (partitions[r['id']]['split'] != 'val', r['id'])):
            ident = row['id']
            split = partitions[ident]['split']
            if split == 'test':
                continue
            category = metadata[ident]['Material']
            label = MMORWAD_MATERIALS.get(category)
            if len(boxes[ident]) != 1 or boxes[ident][0].get('iscrowd'):
                excluded['invalid_single_object_annotation'] += 1
                continue
            category = 'Clear PET' if category.startswith('Polyethylene Terephthalate') and metadata[ident]['Hue'] == 'Clear' else category
            if row['file_name'] not in names:
                excluded['annotation_filename_disagrees_with_archive'] += 1
                continue
            content = incoming.read(names[row['file_name']])
            with Image.open(BytesIO(content)) as raw:
                image = raw.convert('RGB')
            w, h = image.size
            if [w, h] != [row['width'], row['height']]:
                raise ValueError(f'Annotation dimensions mismatch for {ident}')
            x, y, bw, bh = boxes[ident][0]['bbox']
            box = [max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)]
            if min(box[2] - box[0], box[3] - box[1]) < 16:
                excluded['invalid_box'] += 1
                continue
            source_sha = digest(content)
            p, d = perceptual(image.crop(rectangular_box(box, w, h, .04)))
            duplicate_splits = duplicates.matches(p, d)
            if (is_known(image, exact, classification, detection, content) or source_sha in source_hashes or
                    duplicate_splits - {split}):
                excluded['known_or_cross_split_duplicate'] += 1
                continue
            source_hashes.add(source_sha)
            duplicates.add(p, d, split)
            unique_id = 200000 + ident
            group = 'MMoRWaD:' + partitions[ident]['item_name'].strip().casefold()
            source_row = {'id': unique_id, 'split': split, 'group': group, 'source_sha256': source_sha,
                          'source_dataset': 'MMoRWaD', 'source_original_split': split}
            data = jpg(image)
            detector.writestr(f'images/{split}/{unique_id:05d}.jpg', data)
            detector_rows.append(source_row | {
                'size': [w, h], 'sha256': digest(data),
                'pixels_sha256': digest(image.tobytes()),
                'phash': str(imagehash.phash(image)),
                'boxes': [{'annotation_id': unique_id, 'category': category, 'material': label,
                           'xyxy_normalized': [v / s for v, s in zip(box, (w, h, w, h))]}],
            })
            if label is None:
                excluded['composite_material_not_classified'] += 1
                continue
            variants = {'square': crop_box(box, w, h), 'rect': rectangular_box(box, w, h, .12),
                        'tight': rectangular_box(box, w, h, 0)}
            paths = {}
            for view, region in variants.items():
                path = f'{split}/{label}/{unique_id}_{view}.jpg'
                materials.writestr(path, encode(np.asarray(image.crop(region))))
                paths[view] = path
            manifests.append(source_row | {'image_id': unique_id, 'class': label, 'category': category,
                                            'dataset': 'MMoRWaD', 'paths': paths, 'size': [bw, bh]})
        audit = {'source': MMORWAD, 'license': 'CC BY 4.0', 'test_images_opened': 0,
                 'human_annotations': True, 'excluded': excluded,
                 'counts': {split: Counter(row['class'] for row in manifests if row['split'] == split)
                            for split in ('train', 'val')},
                 'detector_counts': Counter(row['split'] for row in detector_rows)}
        for archive, manifest in ((materials, manifests), (detector, detector_rows)):
            archive.writestr('manifest.json', json.dumps(manifest))
            archive.writestr('audit.json', json.dumps(audit, indent=2))
    output.with_suffix('.tmp').replace(output)
    image_output.with_suffix('.tmp').replace(image_output)
    print(json.dumps(audit, indent=2), flush=True)


def bdwaste(source, output):
    exact, classification, detection = new_photo_filter()
    duplicates, seen, counts, excluded = NearPhotos(), set(), Counter(), Counter()
    names = ('Banana peel', 'Egg shell', 'Lemon Peel', 'Potato Peel')
    records = []
    with zipfile.ZipFile(output.with_suffix('.tmp'), 'x') as archive:
        for category in names:
            with zipfile.ZipFile(source / f'{category}.zip') as incoming:
                for member in sorted(incoming.namelist()):
                    if PurePosixPath(member).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                        continue
                    content = incoming.read(member)
                    sha = digest(content)
                    with Image.open(BytesIO(content)) as raw:
                        image = ImageOps.exif_transpose(raw).convert('RGB')
                    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
                    p, d = perceptual(image)
                    pm, dm = perceptual(ImageOps.mirror(image))
                    if (sha in seen or duplicates.matches(p, d) or duplicates.matches(pm, dm) or
                            is_known(image, exact, classification, detection, content)):
                        excluded['exact_or_near_duplicate'] += 1
                        continue
                    ident = f'{category.lower().replace(" ", "-")}-{counts[category]:03d}'
                    path = f'images/{ident}.jpg'
                    picture = jpg(image)
                    archive.writestr(path, picture)
                    seen.add(sha)
                    duplicates.add(p, d, ident)
                    counts[category] += 1
                    records.append({'id': ident, 'material': 'organicos', 'category': category, 'size': image.size,
                                    'path': path, 'sha256': digest(picture), 'source_sha256': sha,
                                    'source_member': member})
        audit = {'source': BDWASTE, 'license': 'CC BY 4.0', 'role': 'external classification only',
                 'counts': counts, 'excluded': excluded}
        archive.writestr('manifest.json', json.dumps(records))
        archive.writestr('audit.json', json.dumps(audit, indent=2))
    output.with_suffix('.tmp').replace(output)
    print(json.dumps(audit, indent=2), flush=True)


def diverse_materials(source, output):
    base = output.parent / 'mmorwad-materials.zip'
    external = output.parent / 'bdwaste-organics.zip'
    exact, classification, detection = new_photo_filter()
    with zipfile.ZipFile(external) as held:
        held_rows = json.loads(held.read('manifest.json'))
        held_categories = {row['category'] for row in held_rows}
        for row in held_rows:
            exact.add(row['source_sha256'])
            with Image.open(BytesIO(held.read(row['path']))) as image:
                p, d = perceptual(image)
                classification.add(p, d, 'external organics')
        held_audit = digest(held.read('audit.json'))
    training_categories = ('Fish ash', 'Mango Peel', 'Rice')
    if set(training_categories) & held_categories:
        raise ValueError('The organic subtypes reserved for external evaluation cannot be used for training.')
    near_duplicates, counts, excluded = NearPhotos(), Counter(), Counter()
    with (zipfile.ZipFile(base) as original,
          zipfile.ZipFile(output.with_suffix('.tmp'), 'x') as archive):
        records = json.loads(original.read('manifest.json'))
        audit = json.loads(original.read('audit.json'))
        exact.update(row['source_sha256'] for row in records)
        for path in original.namelist():
            if path not in ('manifest.json', 'audit.json'):
                archive.writestr(path, original.read(path))
        for category in training_categories:
            with zipfile.ZipFile(source / f'{category}.zip') as incoming:
                for member in sorted(incoming.namelist()):
                    if PurePosixPath(member).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                        continue
                    raw = incoming.read(member)
                    sha = digest(raw)
                    with Image.open(BytesIO(raw)) as image:
                        picture = ImageOps.exif_transpose(image).convert('RGB')
                    picture.thumbnail((384, 384), Image.Resampling.LANCZOS)
                    p, d = perceptual(picture)
                    pm, dm = perceptual(ImageOps.mirror(picture))
                    if (near_duplicates.matches(p, d) or near_duplicates.matches(pm, dm) or
                            is_known(picture, exact, classification, detection, raw)):
                        excluded['known_or_near_duplicate'] += 1
                        continue
                    ident = 600000 + sum(counts.values())
                    path = f'train/organicos/{ident}.jpg'
                    image_bytes = jpg(picture)
                    archive.writestr(path, image_bytes)
                    records.append({'id': ident, 'image_id': ident, 'group': f'BDWaste:{category}',
                                    'split': 'train', 'class': 'organicos', 'dataset': 'BDWaste',
                                    'category': category, 'source_member': member, 'source_sha256': sha,
                                    'sha256': digest(image_bytes), 'path': path})
                    counts[category] += 1
                    exact.add(sha)
                    near_duplicates.add(p, d, ident)
        if not counts or set(counts) != set(training_categories):
            raise ValueError('At least one verified real image is required for each requested organic subtype.')
        audit['counts']['train']['organicos'] = sum(counts.values())
        audit['organic_extension'] = {'source': BDWASTE, 'license': 'CC BY 4.0', 'training_counts': counts,
                                      'excluded': excluded, 'external_subtypes': sorted(held_categories),
                                      'external_audit_sha256': held_audit}
        archive.writestr('manifest.json', json.dumps(records))
        archive.writestr('audit.json', json.dumps(audit, indent=2))
    output.with_suffix('.tmp').replace(output)
    print(json.dumps(audit['organic_extension'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('bower', 'mmorwad', 'bdwaste', 'diverse'))
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--detector', type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.detector and args.detector.exists():
        parser.error('Output already exists.')
    if args.dataset == 'bower':
        bower(args.source, args.output)
    elif args.dataset == 'bdwaste':
        bdwaste(args.source, args.output)
    elif args.dataset == 'diverse':
        diverse_materials(args.source, args.output)
    elif args.detector:
        mmorwad(args.source, args.output, args.detector)
    else:
        parser.error('MMoRWaD requires --detector.')

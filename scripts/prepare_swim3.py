"""Add the published SWiM3 training partition to Vision-IA's detection data."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

import imagehash
from PIL import Image, ImageOps

from scripts.prepare_mwcd import NearPhotos, normalized_photo, protected_photos
from utils.visual_audit import perceptual


REVISION = '1573cfe54241e6ee68590c68d9a26ba5d085f47d'
REPOSITORY = 'Burhan-Ahmed/SWiM3-Dataset'
RAW = f'https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/Splitted_Dataset/train'
# These are the IDs in the authors' data.yaml. Keep the labels for audits only.
MATERIALS = ('plastico', 'papel', 'metal')


def fetch(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'Vision-IA-dataset-import'}), timeout=25) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def tree(folder):
    url = f'https://api.github.com/repos/{REPOSITORY}/contents/{folder}?ref={REVISION}'
    files = json.loads(fetch(url))
    return {row['name']: row['sha'] for row in files}


def read_training_tree(path, kind):
    if path is None:
        url = f'https://api.github.com/repos/{REPOSITORY}/contents/Splitted_Dataset/train?ref={REVISION}'
        directory = next(row for row in json.loads(fetch(url)) if row['name'] == kind and row['type'] == 'dir')
        payload = json.loads(fetch(directory['git_url'] + '?recursive=1'))
    else:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if payload.get('truncated'):
        raise ValueError(f'Incomplete Git tree: {path}')
    return {row['path']: row['sha'] for row in payload['tree'] if row['type'] == 'blob'}


def get_blob(kind, name, expected, cache):
    local = cache / kind / name
    content = local.read_bytes() if local.exists() else fetch(f'{RAW}/{kind}/{quote(name)}')
    git_hash = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
    if git_hash != expected:
        raise ValueError(f'SWiM3 blob hash differs: {name}')
    if not local.exists():
        local.write_bytes(content)
    return content


def read_boxes(content):
    boxes = []
    for line in content.decode('utf-8').splitlines():
        category, x, y, w, h = line.split()
        index = int(category)
        if not 0 <= index < len(MATERIALS):
            raise ValueError(f'Invalid SWiM3 category: {index}')
        x, y, w, h = map(float, (x, y, w, h))
        corners = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        if not all(-.002 <= value <= 1.002 for value in corners) or min(w, h) <= 0:
            raise ValueError('Invalid normalized coordinates.')
        bounds = [max(0., min(1., value)) for value in corners]
        if min(bounds[2] - bounds[0], bounds[3] - bounds[1]) <= 0:
            raise ValueError('Clipped box is empty.')
        boxes.append({'category': MATERIALS[index], 'xyxy_normalized': bounds})
    return boxes


def pack_photo(content):
    with Image.open(BytesIO(content)) as incoming:
        photo = incoming.convert('RGB')
    photo.thumbnail((768, 768), Image.Resampling.LANCZOS)
    output = BytesIO()
    photo.save(output, 'JPEG', quality=88)
    return output.getvalue()


def build(base, image_tree, label_tree, cache, output, workers):
    images, labels = read_training_tree(image_tree, 'images'), read_training_tree(label_tree, 'labels')
    if len(images) != 4992 or len(labels) != 4992:
        raise ValueError('The expected pinned train partition contains 4,992 image/label pairs.')
    reserved = {sha for split in ('valid', 'test') for sha in tree(f'Splitted_Dataset/{split}/images').values()}
    for directory in ('images', 'labels'):
        (cache / directory).mkdir(parents=True, exist_ok=True)
    exact, classification, detection = protected_photos()
    duplicates = NearPhotos()
    filtered, counts, images_by_material = Counter(), Counter(), Counter()

    def download(item):
        name, blob = item
        label_file = PurePosixPath(name).with_suffix('.txt').name
        if blob in reserved:
            return name, None, None, 'publisher_reserved_exact'
        annotation = get_blob('labels', label_file, labels[label_file], cache)
        try:
            boxes = read_boxes(annotation)
        except ValueError:
            return name, None, None, 'invalid_annotation'
        if not boxes:
            return name, None, None, 'empty_annotation'
        content = get_blob('images', name, blob, cache)
        return name, content, boxes, None

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(base) as existing, zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED) as archive:
        source_manifest = existing.read('manifest.json')
        rows = json.loads(source_manifest)
        if {row['split'] for row in rows} != {'train', 'val'}:
            raise ValueError('The detection archive must contain only development partitions.')
        next_id = max(r['id'] for r in rows) + 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (name, content, boxes, reason) in enumerate(pool.map(download, sorted(images.items())), 1):
                if index % 250 == 0:
                    print(f'SWiM3 checked {index}/{len(images)}', flush=True)
                if reason:
                    filtered[reason] += 1
                    continue
                digest = hashlib.sha256(content).hexdigest()
                if digest in exact:
                    filtered['known_exact'] += 1
                    continue
                try:
                    photo = normalized_photo(content)
                except (OSError, ValueError):
                    filtered['invalid_photo'] += 1
                    continue
                p, d = perceptual(photo)
                mirror = ImageOps.mirror(photo)
                mp, md = perceptual(mirror)
                if (classification.matches(p, d) or classification.matches(mp, md)
                        or any(detection.matches(int(str(imagehash.phash(view)), 16)) for view in (photo, mirror))):
                    filtered['known_visual'] += 1
                    continue
                if duplicates.matches(p, d) or duplicates.matches(mp, md):
                    filtered['swim3_visual_duplicate'] += 1
                    continue
                duplicates.add(p, d, name)
                exact.add(digest)
                packed = pack_photo(content)
                rows.append({'id': next_id, 'split': 'train', 'source_dataset': 'SWiM3', 'source_image': name,
                             'source_git_blob': images[name], 'source_photo_sha256': digest,
                             'sha256': hashlib.sha256(packed).hexdigest(), 'boxes': boxes})
                archive.writestr(f'images/train/{next_id:05d}.jpg', packed)
                next_id += 1
                counts.update(b['category'] for b in boxes)
                images_by_material.update({b['category'] for b in boxes})
        if not counts:
            raise ValueError('No usable SWiM3 training annotations.')
        audit = {'dataset': 'SWiM3', 'source': f'https://github.com/{REPOSITORY}/tree/{REVISION}',
                 'usage': 'Research and education per authors README', 'parent_manifest_sha256': hashlib.sha256(source_manifest).hexdigest(),
                 'new_images': len(rows) - len(json.loads(source_manifest)), 'boxes_by_source_class': counts,
                 'images_by_source_class': images_by_material, 'excluded': filtered,
                 'test_images_opened': 0, 'source_split': 'train', 'maximum_added_photo_side': 768,
                 'contains_only_added_images': True}
        archive.writestr('manifest.json', json.dumps(rows))
        archive.writestr('audit.json', json.dumps(audit, indent=2))
    temporary.replace(output)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'cache', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--image-tree', type=Path)
    parser.add_argument('--label-tree', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    options = parser.parse_args()
    build(options.base, options.image_tree, options.label_tree, options.cache, options.output, options.workers)

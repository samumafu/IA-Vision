"""Import MWCD v1 images for training only, excluding protected and duplicate photos."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import zipfile

import imagehash
from PIL import Image, ImageOps

from utils.visual_audit import ALGORITHM, perceptual


SOURCE = 'https://doi.org/10.17632/x863v66fv3.1'
SOURCE_SHA256 = 'f47311ad20d7a0f6015ea6c404c1a72e48adfd1b2e6549c8ee9261f7a10eff1a'
LABELS = dict(plastic='plastico', paper='papel', cardboard='carton', glass='vidrio', metal='metal', organic='organicos')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


class NearPhotos:
    """Seven disjoint chunks guarantee a candidate for Hamming distance <= 6."""

    def __init__(self):
        self.buckets = defaultdict(list)

    @staticmethod
    def keys(p):
        return ((i, (p >> (i * 9)) & (1023 if i == 6 else 511)) for i in range(7))

    def add(self, p, d, value):
        entry = (p, d, value)
        for key in self.keys(p):
            self.buckets[key].append(entry)

    def matches(self, p, d=None):
        return {value for key in self.keys(p) for other, detail, value in self.buckets.get(key, ())
                if (p ^ other).bit_count() <= 6 and (d is None or (d ^ detail).bit_count() <= 8)}


def protected_photos():
    root = Path('data/experiments/B/processed')
    cache = json.loads(Path('data/experiments/visual_cache.json').read_text(encoding='utf-8'))
    if cache['algorithm'] != ALGORITHM:
        raise ValueError('The classification hash cache uses a different algorithm.')
    exact = set(json.loads(Path('exports/protected_classification_hashes.json').read_text())['sha256'])
    classification, detection = NearPhotos(), NearPhotos()
    with (root / 'manifest.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            exact.add(row['sha256'])
            for p, d in cache['images'][row['sha256']]['hashes']:
                classification.add(p, d, 'B')
    for row in json.loads(Path('data/detection/refine_D2/manifest.json').read_text(encoding='utf-8')):
        exact.update(row[key] for key in ('sha256', 'source_png_sha256', 'downloaded_sha256') if row.get(key))
        detection.add(int(row['phash'], 16), 0, 'detection')
    with zipfile.ZipFile('runs/scene_quality/scene_materials.zip') as archive:
        rows = json.loads(archive.read('manifest.json'))
        for row in rows:
            if row['split'] != 'val':
                continue
            for name in ('tight', 'rect', 'square', 'foreground'):
                content = archive.read(row['paths'][name])
                exact.add(hashlib.sha256(content).hexdigest())
                with Image.open(BytesIO(content)) as image:
                    p, d = perceptual(image)
                classification.add(p, d, 'scene validation crop')
    return exact, classification, detection


def normalized_photo(content):
    with Image.open(BytesIO(content)) as raw:
        image = ImageOps.exif_transpose(raw).convert('RGB')
        if min(image.size) < 48:
            raise ValueError('Image is smaller than 48 pixels on one side.')
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
    return image


def build(source, output):
    with source.open('rb') as stream:
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    if source_hash != SOURCE_SHA256:
        raise ValueError('The archive does not match the published MWCD v1 SHA-256.')
    exact, classification, detection = protected_photos()
    accepted, files, excluded, conflicted = {}, Counter(), Counter(), set()
    new_photos = NearPhotos()
    with zipfile.ZipFile(source) as incoming:
        for index, member in enumerate(sorted(incoming.namelist())):
            if index % 5000 == 0:
                print('MWCD members reviewed', index, flush=True)
            path = PurePosixPath(member)
            if path.suffix.lower() not in IMAGE_SUFFIXES or path.parent.name.lower() not in LABELS:
                continue
            label = LABELS[path.parent.name.lower()]
            files[label] += 1
            try:
                content = incoming.read(member)
                digest = hashlib.sha256(content).hexdigest()
                if digest in exact:
                    excluded['known_exact'] += 1
                    continue
                image = normalized_photo(content)
            except (OSError, ValueError, Image.DecompressionBombError):
                excluded['invalid_or_small'] += 1
                continue
            p, d = perceptual(image)
            mirrored = ImageOps.mirror(image)
            p_mirror, d_mirror = perceptual(mirrored)
            if (classification.matches(p, d) or classification.matches(p_mirror, d_mirror)
                    or any(detection.matches(int(str(imagehash.phash(view)), 16)) for view in (image, mirrored))):
                excluded['known_visual'] += 1
                continue
            matches = new_photos.matches(p, d) | new_photos.matches(p_mirror, d_mirror)
            if matches:
                if any(accepted[other]['class'] != label or other in conflicted for other in matches):
                    conflicted.update(matches)
                    excluded['conflicting_visual_label'] += 1
                else:
                    excluded['internal_visual_duplicate'] += 1
                continue
            accepted[member] = {'class': label, 'split': 'train', 'dataset': 'MWCD', 'source_sha256': digest,
                                'path': f'train/{label}/{digest}.jpg', 'source_member': member}
            new_photos.add(p, d, member)
        if set(files) != set(LABELS.values()):
            raise ValueError(f'Unexpected MWCD directories: {files}')
        rows = [row for member, row in accepted.items() if member not in conflicted]
        excluded['removed_for_conflicting_label'] = len(conflicted)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.tmp')
        if output.exists():
            raise FileExistsError(output)
        with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_STORED) as archive:
            for row in rows:
                image = normalized_photo(incoming.read(row['source_member']))
                image.thumbnail((224, 224), Image.Resampling.LANCZOS)
                stream = BytesIO()
                image.save(stream, 'JPEG', quality=90)
                encoded = stream.getvalue()
                row['sha256'] = hashlib.sha256(encoded).hexdigest()
                archive.writestr(row['path'], encoded)
            audit = {'dataset': 'MWCD', 'source': SOURCE, 'license': 'CC BY 4.0', 'source_sha256': source_hash,
                     'original_images': files, 'training_images': Counter(r['class'] for r in rows),
                     'excluded': excluded, 'test_images_opened': 0, 'split': 'train'}
            archive.writestr('manifest.json', json.dumps(rows))
            archive.writestr('audit.json', json.dumps(audit, indent=2))
        temporary.replace(output)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)

"""Build new camera-like training crops and check for overlap with fixed evaluations."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.detector import crop_box
from scripts.prepare_mwcd import NearPhotos
from scripts.prepare_near_camera import digest, new_photo_filter
from scripts.prepare_scene_materials import encode, rectangular_box
from utils.visual_audit import perceptual


SOURCES = {
    'DrinkingWaste': 'https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification',
    'DWSD': 'https://doi.org/10.17632/gr99ny6b8p.1',
    'DWaste': 'https://www.dwaste.live/dataset/',
    'ISEP': 'https://doi.org/10.17632/h5pxbsdz4m.1',
    'TUWien': 'https://doi.org/10.48436/27k90-dvw73',
}
DRINKING = {0: 'metal', 1: 'vidrio', 2: 'plastico', 3: 'plastico'}
DWSD = {1: 'plastico', 2: 'plastico', 4: 'metal', 6: 'vidrio', 8: 'plastico',
        9: 'papel', 10: 'plastico', 12: 'metal'}
DWASTE = {'paper': 'papel', 'cardboard': 'carton', 'glass': 'vidrio', 'biological': 'organicos'}


def image_from(content):
    with Image.open(BytesIO(content)) as image:
        return image.convert('RGB')


def hashes(image, both=False):
    small = image.copy()
    small.thumbnail((256, 256), Image.Resampling.LANCZOS)
    values = [perceptual(small)]
    if both:
        values.append(perceptual(small.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    return values


def image_records(source):
    with zipfile.ZipFile(source / 'drinking.zip') as archive:
        files = set(archive.namelist())
        for name in sorted(files):
            if '/YOLO_imgs/' not in name or not name.endswith('.jpg'):
                continue
            annotation = name[:-4] + '.txt'
            if annotation not in files:
                continue
            lines = archive.read(annotation).splitlines()
            if len(lines) != 1:
                continue
            category, cx, cy, width, height = map(float, lines[0].split())
            content = archive.read(name)
            image = image_from(content)
            w, h = image.size
            box = ((cx - width / 2) * w, (cy - height / 2) * h,
                   (cx + width / 2) * w, (cy + height / 2) * h)
            yield 'DrinkingWaste', name, content, image, [(DRINKING[int(category)], box)]
    with zipfile.ZipFile(source / 'dwsd-data.zip') as archive:
        members = set(archive.namelist())
        for name in sorted(members):
            if '/Train/Image/' not in name or not name.endswith('.png'):
                continue
            mask_name = name.replace('/Image/img_', '/Mask/mask_')
            if mask_name not in members:
                continue
            content = archive.read(name)
            image = image_from(content)
            with Image.open(BytesIO(archive.read(mask_name))) as mask:
                if mask.size != image.size or mask.mode != 'L':
                    continue
                labels = np.asarray(mask)
            objects = []
            for pixel, label in DWSD.items():
                _, _, stats, _ = cv2.connectedComponentsWithStats((labels == pixel).astype('uint8'), connectivity=8)
                for x, y, w, h, area in stats[1:]:
                    if min(w, h) >= 22 and area >= .20 * w * h:
                        region = labels[y:y+h, x:x+w]
                        other = (region != pixel) & (region != 0) & (region != 15)
                        if other.mean() <= .12:
                            objects.append((label, (x, y, x+w, y+h)))
            if objects:
                yield 'DWSD', name, content, image, objects


def reviewed_records(source, selected):
    with zipfile.ZipFile(source / 'dwaste.zip') as archive:
        for name, box in selected.items():
            path = PurePosixPath(name)
            if len(path.parts) != 3 or path.parts[0] != 'original':
                raise ValueError(f'Invalid reviewed DWaste path: {name}')
            label = DWASTE[path.parts[1]]
            content = archive.read(name)
            image = image_from(content)
            w, h = image.size
            yield 'DWaste', name, content, image, [(label, tuple(v*s for v, s in zip(box, (w, h,w,h))))]


def domestic_records(source, accepted):
    mapping = {'Cardboard': 'carton', 'Paper': 'papel', 'Glass': 'vidrio'}
    with zipfile.ZipFile(source / 'isep.zip') as archive:
        for name in sorted(archive.namelist()):
            path = PurePosixPath(name)
            if (name not in accepted or len(path.parts) < 3 or path.parts[-3] != 'Urban and Household Waste' or
                    path.parts[-2] not in mapping or path.suffix.lower() not in ('.jpg', '.jpeg', '.png')):
                continue
            raw = archive.read(name)
            with Image.open(BytesIO(raw)) as incoming:
                image = ImageOps.exif_transpose(incoming).convert('RGB')
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            w, h = image.size
            yield 'ISEP', name, raw, image, [(mapping[path.parts[-2]], (0, 0, w, h))]
    directory = source / 'tuw'
    for path in sorted(directory.glob('2024-*.png')):
        raw = path.read_bytes()
        image = image_from(raw)
        with Image.open(directory / f'mask-{path.name}') as mask:
            selected = np.all(np.asarray(mask.convert('RGB')) == (131, 224, 112), axis=2)
        y, x = np.nonzero(selected)
        if len(x) and x.max() - x.min() >= 30 and y.max() - y.min() >= 30:
            yield 'TUWien', path.name, raw, image, [('organicos', (x.min(), y.min(), x.max()+1, y.max()+1))]


def protected():
    exact, originals, detection = new_photo_filter()
    holdout = NearPhotos()
    evaluation = {}
    for filename in ('bower.zip', 'bdwaste-organics.zip', 'mmorwad-detector.zip'):
        path = Path('runs/near_camera') / filename
        with zipfile.ZipFile(path) as archive:
            manifest = archive.read('manifest.json')
            evaluation[filename] = digest(manifest)
            for row in json.loads(manifest):
                if row.get('split') == 'train':
                    continue
                exact.add(row['source_sha256'])
                member = row.get('path') or f"images/val/{row['id']:05d}.jpg"
                image = image_from(archive.read(member))
                for p,d in hashes(image, both=True):
                    holdout.add(p,d,'photo')
                for obj in row.get('truth', []):
                    for coords in (crop_box(obj['box'], *image.size), rectangular_box(obj['box'], *image.size, 0.)):
                        for p,d in hashes(image.crop(coords), both=True):
                            holdout.add(p,d,'crop')
    known = NearPhotos()
    with zipfile.ZipFile('runs/intensive/mwcd-train.zip') as archive:
        for row in json.loads(archive.read('manifest.json')):
            picture = image_from(archive.read(row['path']))
            p,d = hashes(picture)[0]
            known.add(p,d,'MWCD')
    with zipfile.ZipFile('runs/near_camera/near-materials-diverse.zip') as archive:
        for row in json.loads(archive.read('manifest.json')):
            exact.add(row['source_sha256'])
    return exact, originals, detection, holdout, known, evaluation


def build(source, reviewed, output, domestic=None):
    exact, originals, detection, holdout, known, evaluation = protected()
    seen = NearPhotos()
    rows, excluded = [], Counter()
    from itertools import chain
    import imagehash
    selected = json.loads(reviewed.read_text(encoding='utf-8')) if reviewed else {}
    with zipfile.ZipFile(output.with_suffix('.tmp'), 'x') as archive:
        additional = domestic_records(source, set(json.loads(domestic.read_text(encoding='utf-8')))) if domestic else ()
        for dataset, member, content, image, objects in chain(additional, reviewed_records(source,selected),image_records(source)):
            source_sha = digest(content)
            fingerprints = hashes(image, both=True)
            if (source_sha in exact or any(originals.matches(p,d) or holdout.matches(p,d) for p,d in fingerprints) or
                    detection.matches(int(str(imagehash.phash(image)),16)) or seen.matches(*fingerprints[0])):
                excluded[dataset, 'photo_overlap'] += 1
                continue
            for label, box in objects:
                w,h = image.size
                square = crop_box(box,w,h)
                tight = rectangular_box(box,w,h,0.)
                rect = rectangular_box(box,w,h,.12)
                if min(tight[2]-tight[0],tight[3]-tight[1]) < 20:
                    excluded[dataset, 'too_small'] += 1
                    continue
                crops = {name: image.crop(coords) for name,coords in (('square',square),('tight',tight),('rect',rect))}
                crop_hashes = [descriptor for crop in crops.values() for descriptor in hashes(crop, both=True)]
                if any(originals.matches(p,d) or holdout.matches(p,d) or known.matches(p,d) or seen.matches(p,d)
                       for p,d in crop_hashes):
                    excluded[dataset, 'crop_overlap'] += 1
                    continue
                ident = 700000 + len(rows)
                paths = {view: f'train/{label}/{ident}_{view}.jpg' for view in crops}
                for view, crop in crops.items():
                    archive.writestr(paths[view], encode(np.asarray(crop)))
                rows.append({'dataset': dataset,'source_member': member,'source_sha256': source_sha,'id': ident,
                             'image_id': source_sha,'group': f'{dataset}:{member}', 'split': 'train',
                             'class': label,'paths': paths})
                for p,d in crop_hashes[::2]:
                    seen.add(p,d,'accepted')
            if rows and rows[-1]['source_sha256'] == source_sha:
                seen.add(*fingerprints[0],'accepted')
                exact.add(source_sha)
        counts = Counter(f"{r['dataset']}/{r['class']}" for r in rows)
        audit = {'role': 'camera_training_only','test_images_opened': 0,'protected_evaluations': evaluation,
                 'sources': {name: url for name,url in SOURCES.items() if any(key.startswith(name+'/') for key in counts)},
                 'counts': counts,'excluded': {f'{k[0]}/{k[1]}':v for k,v in excluded.items()},
                 'policy': 'human boxes and masks, or reviewed and author-classified close photos',
                 'crop_geometry': 'app square and rectangular with 12% per side, plus tight; full photo and crop deduplication'}
        archive.writestr('manifest.json',json.dumps(rows))
        archive.writestr('audit.json',json.dumps(audit,indent=2))
    output.with_suffix('.tmp').replace(output)
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('downloads',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--reviewed',type=Path)
    parser.add_argument('--domestic',type=Path, help='Reviewed ISEP filenames and locally staged TU Wien masks')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists.')
    build(args.downloads,args.reviewed,args.output,args.domestic)

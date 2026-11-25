"""Train an own material classifier from random weights on audited development data."""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import csv
import hashlib
from io import BytesIO, TextIOWrapper
import json
import math
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from training.compact import CompactCNN, build_transform
from training.material_encoder import ARCHITECTURE as ENCODER_ARCHITECTURE, MaterialEncoder
from training.material_transformer import ARCHITECTURE as TRANSFORMER_ARCHITECTURE, MaterialTransformer
from utils.config import CLASSES


ORIGINAL_PREFIX = 'Vision-IA/data/experiments/B/processed/'
VIEW_MIX = {'square': .35, 'rect': .30, 'tight': .25, 'foreground': .10}
CAMERA_SOURCES = {'DWSD', 'DrinkingWaste', 'DWaste', 'ISEP', 'TUWien'}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def original_rows(archive):
    with archive.open(ORIGINAL_PREFIX + 'manifest.csv') as stream:
        return list(csv.DictReader(TextIOWrapper(stream, encoding='utf-8')))


def cache_picture(content, limit=236):
    with Image.open(BytesIO(content)) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        return image


class MaterialSet(Dataset):
    def __init__(self, originals, scenes, rows, config, training=False, extra=None, detected=None, detected_rows=None,
                 near=None, camera=None):
        self.rows = rows
        self.training = training
        self.transform = build_transform(config, training)
        self.view_weights = VIEW_MIX | config.get('crop_view_weights', {}) | {'detected': config.get('detected_view_weight', .65)}
        self.images = []
        for row in rows:
            if 'paths' in row:
                views = VIEW_MIX if training else ('square', 'rect', 'tight')
                source = row.get('dataset')
                archive = camera if source in CAMERA_SOURCES else near if source == 'MMoRWaD' else scenes
                images = {v: cache_picture(archive.read(row['paths'][v])) for v in views if v in row['paths']}
                match = (detected_rows or {}).get((row['image_id'], row['id']))
                if match:
                    content = detected.read(match['path'])
                    if digest(content) != match['sha256']:
                        raise ValueError(f"Changed detector crop: {match['path']}")
                    images['detected'] = cache_picture(content)
                self.images.append(images)
            elif row.get('dataset') in CAMERA_SOURCES | {'MWCD', 'BDWaste'}:
                archive = camera if row['dataset'] in CAMERA_SOURCES else extra if row['dataset'] == 'MWCD' else near
                content = archive.read(row['path'])
                if digest(content) != row['sha256']:
                    raise ValueError(f"Image changed: {row['path']}")
                self.images.append(content)
            else:
                content = originals.read(ORIGINAL_PREFIX + row['path'])
                if digest(content) != row['sha256']:
                    raise ValueError(f"Image changed: {row['path']}")
                self.images.append({'original': cache_picture(content)})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        images = self.images[index]
        if isinstance(images, bytes):
            return self.transform(cache_picture(images)), CLASSES.index(self.rows[index]['class'])
        if not self.training and 'square' in images:
            views = torch.stack([self.transform(images[key]) for key in ('square', 'rect', 'tight')])
            return views, CLASSES.index(self.rows[index]['class'])
        key = random.choices(list(images), [self.view_weights[k] for k in images])[0] if 'square' in images else 'original'
        return self.transform(images[key]), CLASSES.index(self.rows[index]['class'])


def scores(actual, predicted):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(matrix, (np.asarray(actual, dtype=np.intp), np.asarray(predicted, dtype=np.intp)), 1)
    support = matrix.sum(axis=1)
    true_positive = matrix.diagonal()
    predicted_support = matrix.sum(axis=0)
    precision = true_positive / np.maximum(predicted_support, 1)
    recall = true_positive / np.maximum(support, 1)
    per_class = 2 * true_positive / np.maximum(support + predicted_support, 1)
    return {'accuracy': float(true_positive.sum() / max(support.sum(), 1)),
            'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
            'macro_recall_present': float(recall.sum() / max(np.count_nonzero(support), 1)),
            'macro_F1': float(per_class.mean()), 'macro_F1_present': float(per_class.sum() / max(np.count_nonzero(support), 1)),
            'per_class_recall': dict(zip(CLASSES, recall.tolist())),
            'per_class_precision': dict(zip(CLASSES, precision.tolist())),
            'per_class_support': dict(zip(CLASSES, support.tolist())),
            'confusion_matrix': matrix.tolist(), 'objects': int(support.sum())}


def balanced_sampler(rows, config, detected_index=None):
    domain = {'SWiM3': 'swim3', 'MMoRWaD': 'near', 'BDWaste': 'near', 'MWCD': 'extra'}
    domain.update(dict.fromkeys(CAMERA_SOURCES, 'camera'))
    groups = [(domain.get(r.get('dataset'), 'scene' if 'paths' in r else 'original'), r['class']) for r in rows]
    counts = Counter(groups)
    focus = config.get('sampling_focus', {})
    caps = {'scene': 50, 'near': 50}
    if config.get('camera_class_minimum'):
        caps['camera'] = config['camera_class_minimum']
    effective = {key: min(count / caps.get(key[0], 1), 1.) * focus.get(key[1], 1.)
                 for key, count in counts.items()}
    totals = Counter()
    for (domain, _), weight in effective.items():
        totals[domain] += weight
    fractions = {'scene': config['scene_fraction'], 'extra': config.get('extra_fraction', 0.),
                 'swim3': config.get('swim3_fraction', 0.), 'near': config.get('near_fraction', 0.),
                 'camera': config.get('camera_fraction', 0.)}
    fractions['original'] = 1 - sum(fractions.values())
    if set(totals) != {k for k, v in fractions.items() if v > 0} or fractions['original'] <= 0:
        raise ValueError('Sampling fractions must correspond to the supplied training domains.')
    detected_index = detected_index or {}
    difficulty = [detected_index.get((r.get('image_id'), r.get('id')), {}).get('sampling_weight', 1.) for r in rows]
    camera_counts = Counter((r['dataset'], r['class']) for r in rows if r.get('dataset') in CAMERA_SOURCES)
    difficulty = [factor / camera_counts[r['dataset'], r['class']] ** .5 if r.get('dataset') in CAMERA_SOURCES else factor
                  for r, factor in zip(rows, difficulty)]
    adjusted_counts = Counter()
    for key, factor in zip(groups, difficulty):
        adjusted_counts[key] += factor
    weights = [fractions[key[0]] * effective[key] * factor / (totals[key[0]] * adjusted_counts[key])
               for key, factor in zip(groups, difficulty)]
    return WeightedRandomSampler(weights, config['epoch_samples'], replacement=True)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    truth, predictions = [], []
    for images, labels in loader:
        truth.extend(labels.tolist())
        if images.ndim == 5:
            outputs = model(images.flatten(0, 1).to(device)).argmax(1).view(len(labels), -1)
        else:
            outputs = model(images.to(device)).argmax(1)[:, None]
        predictions.extend(outputs.cpu().tolist())
    predictions = np.array(predictions)
    names = ('square', 'rect', 'tight') if predictions.shape[1] == 3 else ('original',)
    return {key: scores(truth, predictions[:, index]) for index, key in enumerate(names)}


def save_checkpoint(path, model, config, epoch, report, provenance):
    payload = {'format_version': 1, 'architecture': config['architecture'], 'classes': list(CLASSES),
               'image_size': config['image_size'], 'config': config, 'epoch': epoch,
               'manifest_sha256': provenance['original_manifest_sha256'],
               'model_state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
               'provenance': provenance, 'validation': report}
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def train(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['initialization'] != 'random_from_scratch' or config['test_used'] is not False:
        raise ValueError('Expected random initialization and reserved test')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = torch.device(args.device or ('cpu' if args.smoke else 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is required for this run.')
    torch.set_num_threads(args.threads)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    with ExitStack() as stack:
        original = stack.enter_context(zipfile.ZipFile(args.original))
        scenes = stack.enter_context(zipfile.ZipFile(args.scenes))
        extra = stack.enter_context(zipfile.ZipFile(args.extra)) if args.extra else None
        detected = stack.enter_context(zipfile.ZipFile(args.detected)) if args.detected else None
        near = stack.enter_context(zipfile.ZipFile(args.near)) if args.near else None
        camera = stack.enter_context(zipfile.ZipFile(args.camera_train)) if args.camera_train else None
        b_rows = original_rows(original)
        b_manifest = original.read(ORIGINAL_PREFIX + 'manifest.csv')
        scene_manifest = scenes.read('manifest.json')
        scene_rows = json.loads(scene_manifest)
        scene_audit = json.loads(scenes.read('audit.json'))
        if scene_audit['test_images_opened'] != 0 or {row['split'] for row in scene_rows} != {'train', 'val'}:
            raise ValueError('Incompatible scene partitions')
        for key in ('image_id', 'group', 'source_sha256'):
            if {r[key] for r in scene_rows if r['split'] == 'train'} & {r[key] for r in scene_rows if r['split'] == 'val'}:
                raise ValueError(f'Scene split overlap on {key}')
        b_train, b_val = ([r for r in b_rows if r['split'] == s] for s in ('train', 'val'))
        scene_train, scene_val = ([r for r in scene_rows if r['split'] == s] for s in ('train', 'val'))
        detected_index, detected_hash = {}, None
        if detected:
            manifest_bytes = detected.read('manifest.json')
            detected_hash = digest(manifest_bytes)
            audit = json.loads(detected.read('audit.json'))
            train_annotations = {(r['image_id'], r['id']): r for r in scene_train if r.get('dataset', 'TACO') == 'TACO'}
            if audit['split'] != 'train' or audit['test_images_opened'] != 0 or audit['minimum_IoU'] < .5:
                raise ValueError('Detector crops must be matched to human training annotations only.')
            for r in json.loads(manifest_bytes):
                key = r['image_id'], r['id']
                known = train_annotations.get(key)
                if (known is None or key in detected_index or r['IoU'] < audit['minimum_IoU'] or
                        not 1 <= r.get('sampling_weight', 1) <= 2 or
                        any(r[k] != known[k] for k in ('split', 'source_sha256', 'class'))):
                    raise ValueError('Detector crop has no valid training annotation.')
                detected_index[key] = r
        extra_rows, extra_manifest = [], b''
        if extra:
            extra_manifest = extra.read('manifest.json')
            extra_rows = json.loads(extra_manifest)
            audit = json.loads(extra.read('audit.json'))
            protected = {r['sha256'] for r in b_rows} | {r['source_sha256'] for r in scene_rows}
            if (audit.get('test_images_opened') != 0 or not extra_rows or
                    any(r['split'] != 'train' or r['dataset'] != 'MWCD' or r['source_sha256'] in protected for r in extra_rows)):
                raise ValueError('Additional images must be verified train-only records.')
        near_train, near_val, near_hash = [], [], None
        if near:
            data = near.read('manifest.json')
            near_hash = digest(data)
            rows = json.loads(data)
            audit = json.loads(near.read('audit.json'))
            protected = ({r['sha256'] for r in b_rows} |
                         {r['source_sha256'] for r in scene_rows + extra_rows})
            if (audit.get('test_images_opened') != 0 or audit.get('human_annotations') is not True or
                    {r['split'] for r in rows} != {'train', 'val'} or
                    any((r['dataset'] != 'MMoRWaD' and
                         not (r['dataset'] == 'BDWaste' and r['split'] == 'train' and r['class'] == 'organicos')) or
                        r['source_sha256'] in protected or r['class'] not in CLASSES
                        for r in rows)):
                raise ValueError('Close-object data must use audited, human-labeled development partitions.')
            near_train, near_val = ([r for r in rows if r['split'] == s] for s in ('train', 'val'))
            for key in ('image_id', 'group', 'source_sha256'):
                if {r[key] for r in near_train} & {r[key] for r in near_val}:
                    raise ValueError(f'Close-object split overlap on {key}')
        camera_rows, camera_hash = [], None
        if camera:
            camera_manifest = camera.read('manifest.json')
            camera_hash = digest(camera_manifest)
            camera_rows = json.loads(camera_manifest)
            audit = json.loads(camera.read('audit.json'))
            known = {r['sha256'] for r in b_rows} | {
                r['source_sha256'] for r in scene_rows + extra_rows + near_train + near_val
            }
            if (audit.get('role') != 'camera_training_only' or audit.get('test_images_opened') != 0 or
                    not audit.get('protected_evaluations') or not camera_rows or
                    any(r['split'] != 'train' or r['class'] not in CLASSES or r['dataset'] not in CAMERA_SOURCES or
                        r['source_sha256'] in known for r in camera_rows)):
                raise ValueError('Camera supplementation requires an audited train-only package.')
        if args.smoke:
            swim_train = [r for r in scene_train if r.get('dataset') == 'SWiM3'][:8]
            b_train, b_val, scene_train, scene_val = [r[:8] for r in (b_train, b_val, scene_train, scene_val)]
            scene_train += swim_train
            smoke_keys = {(r['image_id'], r['id']) for r in scene_train}
            detected_index = {key: value for key, value in detected_index.items() if key in smoke_keys}
            extra_rows = extra_rows[:8]
            organic_sample = [r for r in near_train if r['dataset'] == 'BDWaste'][:1]
            near_train, near_val = near_train[:8] + organic_sample, near_val[:8]
            camera_rows = camera_rows[:8]
            config.update(epochs=1, batch_size=8, epoch_samples=16)
        provenance = {'initialization': config['initialization'], 'pretrained': False,
                      'original_manifest_sha256': digest(b_manifest), 'scene_manifest_sha256': digest(scene_manifest),
                      'counts': dict(zip(('original_train', 'original_val', 'scene_train', 'scene_val'),
                                         map(len, (b_train, b_val, scene_train, scene_val)))), 'test_used': False,
                      'GPU': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                      'extra_manifest_sha256': digest(extra_manifest) if extra else None}
        provenance['counts']['extra_train'] = len(extra_rows)
        provenance['counts']['detected_train_views'] = len(detected_index)
        provenance['detected_manifest_sha256'] = detected_hash
        if near:
            provenance['near_manifest_sha256'] = near_hash
            provenance['counts'].update(near_train=len(near_train), near_val=len(near_val))
        if camera:
            provenance['camera_manifest_sha256'] = camera_hash
            provenance['counts']['camera_train'] = len(camera_rows)
        train_rows = b_train + scene_train + extra_rows + near_train + camera_rows
        print('Caching', provenance, flush=True)
        training = MaterialSet(original, scenes, train_rows, config, training=True, extra=extra,
                               detected=detected, detected_rows=detected_index, near=near, camera=camera)
        original_val = MaterialSet(original, scenes, b_val, config)
        scene_validation = MaterialSet(original, scenes, scene_val, config)
        near_validation = MaterialSet(original, scenes, near_val, config, near=near) if near else None
    loaders = (DataLoader(original_val, batch_size=128), DataLoader(scene_validation, batch_size=64))
    near_loader = DataLoader(near_validation, batch_size=64) if near_validation is not None else None
    workers = 0 if args.smoke or device.type == 'cpu' else config.get('workers', 2)
    loader = DataLoader(training, batch_size=config['batch_size'], sampler=balanced_sampler(train_rows, config, detected_index),
                        num_workers=workers, persistent_workers=workers > 0, pin_memory=device.type == 'cuda')
    if config['architecture'] == TRANSFORMER_ARCHITECTURE:
        model = MaterialTransformer(config)
    elif config['architecture'] == ENCODER_ARCHITECTURE:
        model = MaterialEncoder(config)
    else:
        model = CompactCNN(config['image_size'], config['dropout'], config['channels'], config['pooling'])
    model = model.to(device, memory_format=torch.channels_last)
    decay = config.get('ema_decay', 0.)
    ema = deepcopy(model).requires_grad_(False) if decay else None
    provenance['initial_state_sha256'] = digest(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', .0003))
    warmup = config.get('warmup_epochs', 0)
    if warmup:
        def lr_multiplier(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, config['epochs'] - warmup - 1)
            return .06 + .94 * (1 + math.cos(math.pi * progress)) / 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=.00008)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', .035))
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history = []
    best, balanced = -1., -1.
    resume_file = args.output / 'last_training.pth'
    if args.resume:
        state = torch.load(resume_file, map_location=device, weights_only=True)
        keys = ('original_manifest_sha256', 'scene_manifest_sha256', 'extra_manifest_sha256',
                'detected_manifest_sha256', 'initial_state_sha256', 'near_manifest_sha256', 'camera_manifest_sha256')
        old_counts = {'detected_train_views': 0} | state['provenance']['counts']
        if (state['config'] != config or old_counts != provenance['counts'] or
                any(state['provenance'].get(key) != provenance.get(key) for key in keys)):
            raise ValueError('Cannot resume with different configuration, data, or initialization.')
        model.load_state_dict(state['model'])
        if ema is not None:
            ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if state['scaler'] and amp:
            scaler.load_state_dict(state['scaler'])
        history, best, balanced = state['history'], state['best'], state['balanced']
    for epoch in range(len(history) + 1, config['epochs'] + 1):
        # Seed each epoch independently so the sampling continues identically after a restart.
        loader.sampler.generator = torch.Generator().manual_seed(config['seed'] + epoch)
        tick = time.monotonic()
        model.train()
        losses = []
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = criterion(model(images.to(device, non_blocking=True, memory_format=torch.channels_last)),
                                 labels.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss in epoch {epoch}.')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                with torch.no_grad():
                    for observed, current in zip(ema.state_dict().values(), model.state_dict().values()):
                        if observed.is_floating_point():
                            observed.lerp_(current, 1 - decay)
                        else:
                            observed.copy_(current)
            losses.append(float(loss.detach()))
        scheduler.step()
        evaluated = ema if ema is not None else model
        original = evaluate(evaluated, loaders[0], device)['original']
        scene = evaluate(evaluated, loaders[1], device)
        recall_weight = config.get('selection_recall_weight', 0.)
        scene_score = np.mean([(1 - recall_weight) * v['macro_F1_present'] + recall_weight * v['macro_recall_present']
                               for v in scene.values()]).item()
        report = {'epoch': epoch, 'loss': float(np.mean(losses)), 'original': original, 'scene': scene,
                  'scene_score': scene_score, 'seconds': time.monotonic() - tick}
        if near_loader is not None:
            near_metrics = evaluate(evaluated, near_loader, device)
            near_score = np.mean([(1 - recall_weight) * v['macro_F1_present'] + recall_weight * v['macro_recall_present']
                                  for v in near_metrics.values()]).item()
            near_weight = config.get('selection_near_weight', .25)
            scene_score = (1. - near_weight) * scene_score + near_weight * near_score
            report.update(near=near_metrics, near_score=near_score, scene_score=scene_score)
        original_weight = config.get('selection_original_weight', 0.)
        if original_weight:
            scene_score = (1 - original_weight) * scene_score + original_weight * original['macro_F1']
            report['scene_score'] = scene_score
        history.append(report)
        if scene_score > best:
            best = scene_score
            save_checkpoint(args.output / 'best_scene.pth', evaluated, config, epoch, report, provenance)
        if original['macro_F1'] >= config['minimum_original_macro_F1'] and scene_score > balanced:
            balanced = scene_score
            save_checkpoint(args.output / 'best_balanced.pth', evaluated, config, epoch, report, provenance)
        if not args.smoke:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                     'scaler': scaler.state_dict(), 'config': config, 'provenance': provenance, 'history': history,
                     'best': best, 'balanced': balanced, 'ema': ema.state_dict() if ema is not None else None}
            temporary = resume_file.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(resume_file)
        temporary = args.output / 'history.tmp'
        temporary.write_text(json.dumps(history, indent=2), encoding='utf-8')
        temporary.replace(args.output / 'history.json')
        print(epoch, round(original['macro_F1'], 4), round(scene_score, 4), round(report['seconds'], 1), flush=True)
    status = {'completed_epochs': len(history), 'provenance': provenance}
    (args.output / 'completed.json').write_text(json.dumps(status, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--extra', type=Path)
    parser.add_argument('--detected', type=Path)
    parser.add_argument('--near', type=Path)
    parser.add_argument('--camera-train', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/train_scene_materials.json'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--threads', type=int, default=2)
    train(parser.parse_args())

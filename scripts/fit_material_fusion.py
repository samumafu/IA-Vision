"""Fit a small probability fusion on training images and human-labelled D3 train crops."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import zipfile

import torch
from torch.nn import functional as F

from app.vision import InferenceService
from scripts.evaluate_scene_materials import image_from, predict
from scripts.train_scene_materials import ORIGINAL_PREFIX, digest, original_rows
from utils.config import CLASSES


def fit(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    models = {'refined': InferenceService(config['checkpoint']), 'scene': InferenceService(config['material_candidate_checkpoint']),
              'transformer': InferenceService(args.transformer)}
    randomizer = random.Random(94)
    pictures, classes, domain = [], [], []
    with zipfile.ZipFile(args.original) as archive:
        rows = [r for r in original_rows(archive) if r['split'] == 'train']
        for name in CLASSES:
            group = [r for r in rows if r['class'] == name]
            for row in randomizer.sample(group, min(len(group), 225)):
                content = archive.read(ORIGINAL_PREFIX + row['path'])
                if digest(content) != row['sha256']:
                    raise ValueError('Original training image changed.')
                pictures.append(image_from(content))
                classes.append(CLASSES.index(name))
                domain.append('original')
    with zipfile.ZipFile(args.detected) as archive:
        audit = json.loads(archive.read('audit.json'))
        if audit['split'] != 'train' or audit['test_images_opened'] != 0:
            raise ValueError('Calibration requires audited training crops.')
        for row in json.loads(archive.read('manifest.json')):
            if row['split'] != 'train':
                raise ValueError('Calibration includes a non-training crop.')
            content = archive.read(row['path'])
            if digest(content) != row['sha256']:
                raise ValueError('Detector training crop changed.')
            pictures.append(image_from(content))
            classes.append(CLASSES.index(row['class']))
            domain.append('scene')
    if args.near:
        with zipfile.ZipFile(args.near) as archive:
            audit = json.loads(archive.read('audit.json'))
            if audit.get('test_images_opened') != 0 or audit.get('human_annotations') is not True:
                raise ValueError('Close-object calibration requires human-labeled development data.')
            rows = [r for r in json.loads(archive.read('manifest.json')) if r['split'] == 'train']
            for name in CLASSES:
                group = [r for r in rows if r['class'] == name]
                for row in randomizer.sample(group, min(len(group), 120)):
                    path = row['paths']['square'] if 'paths' in row else row['path']
                    content = archive.read(path)
                    if row.get('sha256') and digest(content) != row['sha256']:
                        raise ValueError('Close-object calibration image changed.')
                    pictures.append(image_from(content))
                    classes.append(CLASSES.index(name))
                    domain.append('near')
    if args.camera_train:
        with zipfile.ZipFile(args.camera_train) as archive:
            audit = json.loads(archive.read('audit.json'))
            rows = json.loads(archive.read('manifest.json'))
            if (audit.get('role') != 'camera_training_only' or audit.get('test_images_opened') != 0 or
                    not audit.get('protected_evaluations') or any(r['split'] != 'train' for r in rows)):
                raise ValueError('Camera calibration requires the audited train-only package.')
            for name in CLASSES:
                group = [r for r in rows if r['class'] == name]
                for row in randomizer.sample(group, min(len(group), 170)):
                    pictures.append(image_from(archive.read(row['paths']['square'])))
                    classes.append(CLASSES.index(name))
                    domain.append('camera')
    domain_sizes = Counter(domain)
    print('Training samples for calibration:', domain_sizes, flush=True)
    probabilities = predict(models, pictures)
    scene_weight = config['material_candidate_weight']
    cnn = torch.tensor((1 - scene_weight) * probabilities['refined'] + scene_weight * probabilities['scene'])
    tx = torch.tensor(probabilities['transformer'])
    inputs = torch.cat((cnn.clamp_min(1e-6).log(), tx.clamp_min(1e-6).log()), dim=1)
    keys = list(zip(domain, classes))
    counts = Counter(keys)
    focus = dict((name, float(weight)) for name, weight in (value.split('=', 1) for value in args.focus))
    if any(name not in CLASSES or not .1 <= weight <= 10 for name, weight in focus.items()):
        raise ValueError('Calibration focus must use known materials and positive weights.')
    raw_weights = [max(counts[key], 15) ** -args.balance * focus.get(CLASSES[key[1]], 1.) for key in keys]
    totals = Counter()
    for kind, weight in zip(domain, raw_weights):
        totals[kind] += weight
    fractions = {'scene': args.scene_fraction, 'near': args.near_fraction,
                 'camera': args.camera_fraction,
                 'original': 1 - args.scene_fraction - args.near_fraction - args.camera_fraction}
    weights = torch.tensor([fractions[kind] * weight / totals[kind] for kind, weight in zip(domain, raw_weights)])
    classes = torch.tensor(classes)
    reference = torch.cat((.75 * torch.eye(len(CLASSES)), .25 * torch.eye(len(CLASSES))), dim=0)
    coefficients = torch.nn.Parameter(reference.clone())
    bias = torch.nn.Parameter(torch.zeros(len(CLASSES)))
    optimizer = torch.optim.AdamW((coefficients, bias), lr=.025, weight_decay=0.)
    for _ in range(220):
        optimizer.zero_grad(set_to_none=True)
        loss = (F.cross_entropy(inputs @ coefficients + bias, classes, reduction='none') * weights).sum()
        loss += .12 * (coefficients - reference).square().sum() + .18 * bias.square().sum()
        loss.backward()
        optimizer.step()
    payload = {'classes': CLASSES, 'matrix': coefficients.detach().tolist(), 'bias': bias.detach().tolist(),
               'checkpoints': {k: v.checkpoint_sha256 for k, v in models.items()}, 'train_counts': domain_sizes,
               'class_balance_exponent': args.balance, 'scene_weight': scene_weight, 'class_weights': focus,
               'scene_training_fraction': args.scene_fraction, 'near_training_fraction': args.near_fraction,
               'camera_training_fraction': args.camera_fraction,
               'test_used': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/app.json'))
    parser.add_argument('--original', type=Path, default=Path('exports/sync/B/Vision-IA_Colab_B.zip'))
    parser.add_argument('--detected', type=Path, required=True)
    parser.add_argument('--near', type=Path)
    parser.add_argument('--camera-train', type=Path)
    parser.add_argument('--transformer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--balance', type=float, default=0.)
    parser.add_argument('--scene-fraction', type=float, default=.65)
    parser.add_argument('--near-fraction', type=float, default=0.)
    parser.add_argument('--camera-fraction', type=float, default=0.)
    parser.add_argument('--focus', nargs='*', default=[])
    args = parser.parse_args()
    if (args.scene_fraction <= 0 or min(args.near_fraction, args.camera_fraction) < 0 or
            args.scene_fraction + args.near_fraction + args.camera_fraction >= 1 or
            bool(args.near) != (args.near_fraction > 0) or
            bool(args.camera_train) != (args.camera_fraction > 0)):
        parser.error('Fractions must keep original and scene samples and agree with the supplied close-object data.')
    fit(args)

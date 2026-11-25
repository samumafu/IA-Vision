"""Stage verified development archives in Colab and train isolated close-object candidates."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage(source, destination):
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    temporary = destination.with_suffix('.tmp')
    shutil.copyfile(source, temporary)
    with zipfile.ZipFile(temporary) as archive:
        if not archive.namelist():
            raise ValueError(f'Empty archive: {source}')
    temporary.replace(destination)


def stage_detector(work, root):
    destination = work / 'detector_data'
    if (destination / 'audit.json').is_file():
        return destination
    destination.mkdir(exist_ok=True)
    sources = {'base': root / 'scene_quality/detector_trainval.zip',
               'swim3': root / 'intensive/swim3-extra.zip',
               'near': root / 'near_camera/mmorwad-detector.zip'}
    manifests, audits = {}, {}
    for key, source in sources.items():
        local = work / f'detector_{key}.zip'
        stage(source, local)
        with zipfile.ZipFile(local) as archive:
            manifests[key] = json.loads(archive.read('manifest.json'))
            audits[key] = hashlib.sha256(archive.read('audit.json')).hexdigest()
            for member in archive.namelist():
                if member.startswith(('images/train/', 'images/val/')):
                    archive.extract(member, destination)
    original, swim, near = (manifests[key] for key in sources)
    if ([r for r in swim if r.get('source_dataset') != 'SWiM3'] != original or
            any(r['split'] != 'train' for r in swim if r.get('source_dataset') == 'SWiM3') or
            {r['split'] for r in near} != {'train', 'val'}):
        raise ValueError('The extension changed the original detector partitions.')
    merged = swim + near
    if len({r['id'] for r in merged}) != len(merged):
        raise ValueError('Duplicate detector image identifiers.')
    for key in ('group', 'sha256', 'pixels_sha256', 'source_photo_sha256', 'source_sha256'):
        training_values = {r[key] for r in merged if r['split'] == 'train' and r.get(key)}
        held_values = {r[key] for r in merged if r['split'] != 'train' and r.get(key)}
        if training_values & held_values:
            raise ValueError(f'Detector partition overlap on {key}')
    (destination / 'manifest.json').write_text(json.dumps(merged), encoding='utf-8')
    audit = {'source_audit_sha256': audits, 'test_images_opened': 0, 'original_rows_unchanged': True,
             'counts': Counter(r['split'] for r in merged)}
    (destination / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return destination


def run(shared, work, args, log):
    command = [sys.executable, '-u', '-m', *map(str, args)]
    with (shared / log).open('a') as stream:
        subprocess.run(command, cwd=work, stdout=stream, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})


def main(root, work, experiment):
    work.mkdir(parents=True, exist_ok=True)
    shared = root / 'near_camera'
    material_run = experiment in ('materials', 'materials_diverse')
    if material_run:
        near_file = 'mmorwad-materials.zip' if experiment == 'materials' else 'near-materials-diverse.zip'
        inputs = {'original': sorted((root / 'packages/B').glob('*/Vision-IA_Colab_B.zip'))[0],
                  'scenes': root / 'intensive/scene-materials-swim3.zip',
                  'extra': root / 'intensive/mwcd-train.zip',
                  'detected': root / 'transformer/d3_hard_train_crops.zip',
                  'near': shared / near_file}
        flags = ['scripts.train_scene_materials', '--config', 'config/train_near_materials.json']
        for name, source in inputs.items():
            local = work / (near_file if name == 'near' else f'material_{name}.zip')
            stage(source, local)
            flags += [f'--{name}', local]
        smoke, output = work / f'{experiment}_smoke', shared / experiment
        if not (smoke / 'completed.json').is_file():
            run(shared, work, flags + ['--smoke', '--device', 'cuda', '--output', smoke], f'{experiment}-smoke.log')
        if not (output / 'completed.json').is_file():
            resume = ['--resume'] if (output / 'last_training.pth').is_file() else []
            run(shared, work, flags + ['--output', output] + resume, f'{experiment}.log')
    else:
        root_dir = stage_detector(work, root)
        flags = ['scripts.train_detector_own', '--config', 'config/detector_D3_near.json', '--root', root_dir]
        detector_runs = shared / 'detector'
        smoke, output = detector_runs / 'smoke', detector_runs / 'near_D3'
        if not (smoke / 'smoke.json').is_file():
            run(shared, work, flags + ['--smoke', '--output', smoke], 'detector-smoke.log')
        status = detector_runs / 'status.json'
        if not (status.is_file() and json.loads(status.read_text(encoding='utf-8'))['complete']):
            resume = ['--resume', output / 'last.pth'] if (output / 'last.pth').is_file() else []
            run(shared, work, flags + ['--output', output] + resume, 'detector.log')
    names = ('best_scene.pth', 'best_balanced.pth') if material_run else ('best.pth',)
    summary = {name: sha256(output / name) for name in names if (output / name).is_file()}
    (shared / f'{experiment}-checkpoints.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(experiment, summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=('materials', 'materials_diverse', 'detector'))
    parser.add_argument('--root', type=Path, default=Path('/content/drive/MyDrive/Vision-IA'))
    parser.add_argument('--work', type=Path, default=Path('/content/vision_near_camera'))
    args = parser.parse_args()
    main(args.root, args.work, args.experiment)

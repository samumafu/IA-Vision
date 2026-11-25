"""Stage the unchanged development splits and run the camera-data ablation on CUDA."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.train_near_camera_colab import sha256, stage


def train(root, work, experiment):
    shared = root / 'external_materials'
    work.mkdir(parents=True, exist_ok=True)
    inputs = {'original': sorted((root / 'packages/B').glob('*/Vision-IA_Colab_B.zip'))[0],
              'scenes': root / 'intensive/scene-materials-swim3.zip',
              'extra': root / 'intensive/mwcd-train.zip',
              'detected': root / 'transformer/d3_hard_train_crops.zip',
              'near': root / 'near_camera/near-materials-diverse.zip'}
    if experiment in ('camera_data', 'domestic_data'):
        filename = 'camera-training.zip' if experiment == 'camera_data' else 'domestic-training.zip'
        inputs['camera-train'] = shared / filename
    flags = [sys.executable, '-u', '-m', 'scripts.train_scene_materials']
    # Experiment-specific paths keep interrupted copies independent between jobs.
    for name, source in inputs.items():
        path = work / f'{experiment}-{name}.zip'
        stage(source, path)
        flags += [f'--{name}', str(path)]
    config = json.loads((work / 'config/train_external_materials.json').read_text())
    if experiment == 'control':
        config.update(camera_fraction=0., extra_fraction=.35)
    if experiment == 'domestic_data':
        config.update(camera_class_minimum=70, camera_fraction=.36, extra_fraction=.13)
    path = work / f'{experiment}.json'
    path.write_text(json.dumps(config, indent=2))
    flags += ['--config', str(path)]
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    smoke = work / f'{experiment}_smoke'
    if not (smoke / 'completed.json').is_file():
        subprocess.run(flags + ['--output', str(smoke), '--device', 'cuda', '--smoke'], cwd=work, env=env, check=True)
    output = shared / experiment
    if not (output / 'completed.json').is_file():
        resume = ['--resume'] if (output / 'last_training.pth').is_file() else []
        subprocess.run(flags + ['--output', str(output)] + resume, cwd=work, env=env, check=True)
    result = {name: sha256(output / name) for name in ('best_scene.pth', 'best_balanced.pth')
              if (output / name).exists()}
    (shared / f'{experiment}-checkpoints.json').write_text(json.dumps(result, indent=2))
    print(experiment, result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=('control', 'camera_data', 'domestic_data'))
    parser.add_argument('--root', type=Path, default=Path('/content/drive/MyDrive/Vision-IA'))
    parser.add_argument('--work', type=Path, default=Path('/content/vision_external_materials'))
    args = parser.parse_args()
    train(args.root, args.work, args.experiment)

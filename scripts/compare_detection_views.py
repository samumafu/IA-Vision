"""Test whether agreement across views improves D2 ranking without changing its weights."""
import argparse
import itertools
import json
from pathlib import Path

from scripts.diagnose_scene_detector import operating_points
from training.detection_metrics import fuse_detection_views


def compare(args):
    predictions = {}
    for source in args.predictions:
        predictions.update(json.loads(source.read_text(encoding='utf-8')))
    candidates = {}
    report = {}
    groups = [tuple(item.split(',')) for item in args.views] if args.views else [
        ('512', '512flip'), ('512', '640'), ('512', '640', '512flip'), ('512', '640', '512flip', '640flip')]
    for group, factor in itertools.product(groups, (.62, .78, .90)):
        name = '_'.join(group) + f'_lone{factor:g}'
        rows = []
        for source_rows in zip(*(predictions[key] for key in group), strict=True):
            if len({row['image_id'] for row in source_rows}) != 1:
                raise ValueError('Views disagree on source photo')
            fused = fuse_detection_views([r['predictions'] for r in source_rows], lone_factor=factor)
            rows.append(dict(source_rows[0], predictions=fused))
        report[name] = operating_points(rows)
        candidates[name] = rows
        print(name, round(report[name]['mAP50'], 4), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.output / 'predictions.json').write_text(json.dumps(candidates), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, nargs='+', required=True)
    parser.add_argument('--views', nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    compare(parser.parse_args())

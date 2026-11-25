"""Package byte-verified train/val photos with a bounded decode size for Colab."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

from PIL import Image

from scripts.diagnose_scene_detector import load_scene, validation_rows


def build(root, output):
    validation_rows(root)
    source = (root / 'manifest.json').read_bytes()
    rows = [r for r in json.loads(source) if r['split'] in ('train', 'val')]
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_STORED) as archive:
        for row in rows:
            rgb = load_scene(root, row)
            picture = Image.fromarray(rgb)
            picture.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            stream = BytesIO()
            picture.save(stream, format='JPEG', quality=94)
            data = stream.getvalue()
            row['source_photo_sha256'] = row['sha256']
            row['sha256'] = hashlib.sha256(data).hexdigest()
            archive.writestr(f"images/{row['split']}/{row['id']:05d}.jpg", data)
        archive.writestr('manifest.json', json.dumps(rows))
        archive.writestr('audit.json', json.dumps({'source_manifest_sha256': hashlib.sha256(source).hexdigest(),
                                                  'maximum_side': 1024, 'labels_changed': False,
                                                  'test_images_opened': 0, 'images': len(rows)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(Path('data/detection/refine_D2'), args.output)

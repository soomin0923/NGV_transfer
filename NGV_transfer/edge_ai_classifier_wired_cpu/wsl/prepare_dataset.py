"""Validate camera captures and create a session-held-out train/val/test split.

The same course_config.json must be used for collection, this split and training.
No shuffled frame-level split: near-neighbour frames from a session stay together.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import shutil
import tempfile
from PIL import Image
from extra_common.course import load_config
from extra_common.util import write_json, sha256

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def prepare_dataset(src, dst, config):
    src, dst = Path(src), Path(dst)
    classes = list(config['labels'])
    sessions = list(config['sessions'])
    groups = {split:list(config[f'{split}_sessions']) for split in ('train', 'val', 'test')}
    count = config['photos_per_session']
    if dst.exists():
        raise ValueError(f'{dst} already exists. Keep it as a previous run or choose another --output; do not merge splits.')
    if not src.is_dir():
        raise ValueError(f'Missing capture folder: {src}')
    if not classes or len(classes) != len(set(classes)):
        raise ValueError('Empty or duplicate configured labels')
    assigned = [s for values in groups.values() for s in values]
    if any(not values for values in groups.values()) or len(assigned) != len(set(assigned)) or set(assigned) != set(sessions):
        raise ValueError('train/val/test sessions must be nonempty, disjoint, and cover every configured session')
    if not isinstance(count, int) or count < 1:
        raise ValueError('photos_per_session must be a positive integer')
    all_images = {p.resolve() for p in src.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    seen_images, seen_hashes, manifest = set(), {}, []
    src_resolved = src.resolve()
    for cls in classes:
        for split, split_sessions in groups.items():
            for session in split_sessions:
                folder = src / cls / session
                images = sorted(p for p in folder.glob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
                if len(images) != count:
                    raise ValueError(f'{folder}: {len(images)} photos; expected {count} from course_config.json')
                for path in images:
                    resolved = path.resolve()
                    if not resolved.is_relative_to(src_resolved):
                        raise ValueError(f'Image points outside the capture folder: {path}')
                    if resolved in seen_images:
                        raise ValueError(f'Image reused via a symbolic link: {path}')
                    seen_images.add(resolved)
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        image.convert('RGB').load()
                        if min(image.size) < 8:
                            raise ValueError(f'Image is too small: {path}')
                    digest = sha256(path)
                    previous = seen_hashes.get(digest)
                    if previous is not None and previous['split'] != split:
                        raise ValueError(f'Identical image across held-out splits: {previous["source"]} and {path}. Recapture different views in each session.')
                    row = {'source':str(path), 'split':split, 'class':cls, 'session':session, 'sha256':digest}
                    seen_hashes[digest] = row
                    manifest.append(row)
    unexpected = all_images - seen_images
    if unexpected:
        example = '\n'.join(str(p) for p in sorted(unexpected)[:5])
        raise ValueError('Unexpected image files outside configured class/session folders:\n' + example)
    dst.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=dst.name + '.building-', dir=dst.parent))
    try:
        for row in manifest:
            target = staging / row['split'] / row['class'] / (row['session'] + '_' + Path(row['source']).name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row['source'], target)
        (staging / 'class_names.txt').write_text('\n'.join(classes) + '\n', encoding='utf-8')
        write_json(staging / 'split_manifest.json', manifest)
        write_json(staging / 'dataset_config.json', {'labels':classes, 'photos_per_session':count, 'sessions':sessions,
                   **{f'{k}_sessions':v for k, v in groups.items()}})
        if dst.exists():
            raise ValueError(f'{dst} was created by another process; refusing to overwrite')
        staging.rename(dst)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for split in groups:
        print(split, sum(row['split'] == split for row in manifest))
    print('PASS: session-held-out split. Label/output index order:', classes)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='dataset_raw')
    p.add_argument('--output', default='dataset_split')
    p.add_argument('--config', default=None, help='Course JSON; default: project-root course_config.json')
    a = p.parse_args()
    try:
        prepare_dataset(a.input, a.output, load_config(a.config))
    except (ValueError, OSError) as exc:
        raise SystemExit(f'STOP: {exc}') from exc


if __name__ == '__main__':
    main()

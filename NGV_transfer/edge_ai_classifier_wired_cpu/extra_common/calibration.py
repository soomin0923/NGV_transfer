"""Deterministic, balanced training-only calibration image selection."""
from pathlib import Path
import json
from collections import Counter
import numpy as np
from PIL import Image
from extra_common.util import sha256

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def image_files(folder):
    return sorted(p for p in Path(folder).glob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)


def load_image(path, height, width):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((width, height), Image.Resampling.BILINEAR), np.float32)


def calibration_paths(folder, names, limit):
    """Balanced, deterministic coverage across sorted sessions; verify provenance."""
    folder = Path(folder).resolve()
    if folder.name != 'train' or not folder.is_dir():
        raise ValueError('--calibration-dir must be the train directory created by prepare_dataset.py, never val/test.')
    if limit < len(names):
        raise ValueError('--calibration-images must be at least the number of labels.')
    buckets = [image_files(folder / name) for name in names]
    if any(not files for files in buckets):
        raise ValueError('Every configured label must have training calibration images.')
    eligible = [p for bucket in buckets for p in bucket]
    if any(not p.resolve().is_relative_to(folder) for p in eligible):
        raise ValueError('Calibration image symbolic link points outside train.')
    # Check every eligible image, not merely selected ones, against split provenance.
    hashes = {p: sha256(p) for p in eligible}
    manifest_path = folder.parent / 'split_manifest.json'
    manifest_verified = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        train_map = {(r['class'], r['session'] + '_' + Path(r['source']).name): r['sha256']
                     for r in manifest if r['split'] == 'train'}
        heldout = {r['sha256'] for r in manifest if r['split'] in ('val', 'test')}
        for p, digest in hashes.items():
            if train_map.get((p.parent.name, p.name)) != digest:
                raise ValueError(f'Calibration file is absent from / changed since training manifest: {p}')
            if digest in heldout:
                raise ValueError(f'Calibration image also occurs in a held-out split: {p}')
        manifest_verified = True
    else:
        heldout = {sha256(p) for split in ('val', 'test') for name in names
                   for p in image_files(folder.parent / split / name)}
        if heldout.intersection(hashes.values()):
            raise ValueError('Calibration shares identical images with val/test.')
        print('NOTE: split_manifest.json is absent. Train directory and available val/test hashes checked; capture-session provenance is unverified.')
    # Round-robin quotas then evenly spread each quota through all training sessions.
    quotas = [0] * len(names)
    for _ in range(min(limit, len(eligible))):
        available = [i for i in range(len(names)) if quotas[i] < len(buckets[i])]
        i = min(available, key=lambda j: (quotas[j], j))
        quotas[i] += 1
    selected = []
    for bucket, quota in zip(buckets, quotas):
        selected += [bucket[int(i)] for i in np.linspace(0, len(bucket)-1, quota, dtype=int)]
    report = {'source_split': 'train', 'directory': str(folder), 'requested_images': limit,
              'selected_images': len(selected), 'class_counts': dict(Counter(p.parent.name for p in selected)),
              'selection': 'deterministic balanced quotas; evenly spaced sorted filenames per class',
              'split_manifest_verified': manifest_verified,
              'files': [{'path': str(p), 'label': p.parent.name, 'sha256': hashes[p]} for p in selected]}
    return selected, report



def load_training_calibration(train_dir, image_size=(160, 160), max_images=128, names=None):
    """Return float32 RGB [0,255] images and a JSON-serializable provenance dict.

    names is optional; the parent class_names.txt determines label order when
    present, otherwise nonempty image subdirectories are sorted.
    """
    train_dir = Path(train_dir)
    if names is None:
        label_file = train_dir.parent / 'class_names.txt'
        if label_file.exists():
            names = [v.strip() for v in label_file.read_text(encoding='utf-8').splitlines() if v.strip()]
        else:
            names = sorted(d.name for d in train_dir.iterdir() if d.is_dir() and image_files(d))
    if not names or len(names) != len(set(names)):
        raise ValueError('Calibration labels must be nonempty and unique.')
    chosen, report = calibration_paths(train_dir, names, max_images)
    h, w = map(int, image_size)
    if h < 1 or w < 1:
        raise ValueError('image_size must be positive (height, width)')
    return np.stack([load_image(f, h, w) for f in chosen]), report

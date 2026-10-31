"""Local Fashion-MNIST loading and deterministic stratified splits (no network)."""
from pathlib import Path
import numpy as np


def load_local(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing {path}. Prepare fashion-mnist.npz before class and copy it to '
            'offline/data/. This script never downloads data.')
    with np.load(path, allow_pickle=False) as z:
        required = {'x_train', 'y_train', 'x_test', 'y_test'}
        if not required.issubset(z.files):
            raise ValueError(f'{path}: expected keys {sorted(required)}')
        data = {k: z[k].copy() for k in required}
    for split, expected in [('train', 60000), ('test', 10000)]:
        x, y = data['x_' + split], data['y_' + split]
        if x.shape != (expected, 28, 28) or y.shape != (expected,):
            raise ValueError(f'Expected complete official {split} set: '
                             f'[{expected},28,28] images and [{expected}] labels; got {x.shape}, {y.shape}')
        if x.dtype != np.uint8:
            raise ValueError('Images must be original uint8 pixels in [0,255], not normalized arrays.')
        if not np.issubdtype(y.dtype, np.integer) or set(np.unique(y)) != set(range(10)):
            raise ValueError('Fashion-MNIST labels must contain integer classes 0..9.')
        if not np.all(np.bincount(y, minlength=10) == expected // 10):
            raise ValueError(f'Expected balanced complete official {split} labels.')
    return data


def split_indices(y_train, y_test, train_samples=12000, val_samples=3000,
                  test_samples=3000, seed=42, full_data=False):
    """Train/validation use original training indices; evaluation uses original test indices."""
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    if full_data:
        train_samples, val_samples, test_samples = 54000, 6000, 10000
    for name, n in [('train_samples', train_samples), ('val_samples', val_samples),
                    ('test_samples', test_samples)]:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 10 or n % 10:
            raise ValueError(f'{name} must be a positive multiple of 10 (at least 10).')
    rng = np.random.default_rng(seed)
    parts = {'train': [], 'val': [], 'test': []}
    for c in range(10):
        train_pool = rng.permutation(np.flatnonzero(y_train == c))
        test_pool = rng.permutation(np.flatnonzero(y_test == c))
        nt, nv, ne = train_samples // 10, val_samples // 10, test_samples // 10
        if len(train_pool) < nt + nv or len(test_pool) < ne:
            raise ValueError(f'Insufficient class {c} data for requested sample counts.')
        parts['train'].append(train_pool[:nt])
        parts['val'].append(train_pool[nt:nt + nv])
        parts['test'].append(test_pool[:ne])
    result = {}
    for name, arrays in parts.items():
        result[name] = np.concatenate(arrays).astype(np.int64)
        rng.shuffle(result[name])
    if np.intersect1d(result['train'], result['val']).size:
        raise RuntimeError('Training/validation overlap detected')
    return result

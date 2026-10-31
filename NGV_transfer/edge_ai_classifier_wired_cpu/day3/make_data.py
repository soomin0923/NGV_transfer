"""Generate more independent synthetic sessions without changing signal physics.

Default: 4 classes x 150 sessions x 40 windows = 24,000 feature vectors.
The original 30-session, 40-window recipe is available through CLI arguments.
No model, prediction, accuracy target, or test-label feedback is used here.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from day3.numerics import signal, features, LABELS, FS, N
from extra_common.util import ROOT, write_json, finite

SPLITS = ('train', 'val', 'test')


def generate_arrays(seed: int = 42, sessions_per_class: int = 150,
                    windows_per_session: int = 40) -> tuple[dict, dict]:
    """Retain 60/20/20 session splits, the original RNG order and normalization.

    A session's frequency/gain/noise are shared by its windows. Sessions never
    cross splits. Labels and session IDs are metadata, never input features.
    """
    for name, value in [('seed', seed), ('sessions_per_class', sessions_per_class),
                        ('windows_per_session', windows_per_session)]:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f'{name} must be an integer')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    if sessions_per_class < 5 or sessions_per_class % 5:
        raise ValueError('sessions_per_class must be a positive multiple of 5 (minimum 5)')
    if windows_per_session < 1:
        raise ValueError('windows_per_session must be positive')
    n_train = sessions_per_class * 3 // 5
    n_val = sessions_per_class // 5
    n_test = sessions_per_class - n_train - n_val
    rng = np.random.default_rng(seed)
    data = {s: [] for s in SPLITS}
    ys = {s: [] for s in SPLITS}
    ids = {s: [] for s in SPLITS}
    sessions = []
    raw_examples = {s: [] for s in SPLITS}
    example_rows = {s: [] for s in SPLITS}
    example_counts = {(s, k): 0 for s in SPLITS for k in range(len(LABELS))}
    for kind in range(len(LABELS)):
        for session in range(sessions_per_class):
            split = ('train' if session < n_train else
                     'val' if session < n_train + n_val else 'test')
            # Unchanged ranges and call order from the uploaded generator.
            frequency = rng.uniform(34, 46)
            gain = rng.uniform(.8, 1.2)
            noise = rng.uniform(.04, .12)
            session_id = kind * sessions_per_class + session
            sessions.append(dict(session_id=session_id, class_index=kind,
                                 class_name=LABELS[kind], split=split,
                                 frequency_hz=float(frequency), gain=float(gain),
                                 noise_std=float(noise), windows=windows_per_session))
            for _ in range(windows_per_session):
                raw = signal(kind, rng, frequency, gain, noise)
                # Keep the first two real generated windows per class/split.
                # No extra RNG calls: features/labels/sessions remain bit-identical.
                if example_counts[(split, kind)] < 2:
                    raw_examples[split].append(raw.copy())
                    example_rows[split].append(len(data[split]))
                    example_counts[(split, kind)] += 1
                data[split].append(features(raw))
                ys[split].append(kind)
                ids[split].append(session_id)
    # Exactly the original training-only statistics; never fit on val/test.
    mean = np.mean(data['train'], axis=0)
    std = np.maximum(np.std(data['train'], axis=0), 1e-5)
    arrays = {'mean': finite(mean), 'std': finite(std)}
    for split in SPLITS:
        arrays['x_' + split] = finite(
            ((np.stack(data[split]) - mean) / std).astype(np.float32), split)
        arrays['y_' + split] = np.asarray(ys[split], dtype=np.int64)
        arrays['session_' + split] = np.asarray(ids[split], dtype=np.int64)
        arrays['raw_examples_' + split] = np.stack(raw_examples[split]).astype(np.float32)
        arrays['example_rows_' + split] = np.asarray(example_rows[split], dtype=np.int64)
    session_sets = [set(arrays['session_' + s].tolist()) for s in SPLITS]
    if any(session_sets[i] & session_sets[j] for i in range(3) for j in range(i)):
        raise RuntimeError('Session leakage detected')
    description = dict(
        seed=seed, fs_hz=FS, samples_per_window=N, input_features=256,
        sessions_per_class=sessions_per_class, windows_per_session=windows_per_session,
        sessions_per_class_by_split=dict(zip(SPLITS, [n_train, n_val, n_test])),
        train_val_test=[len(arrays['y_' + s]) for s in SPLITS], labels=LABELS,
        synthetic_only=True, normalization='train only',
        signal_recipe='unchanged day3/numerics.py; educational synthetic signals, not real equipment',
        change='increase independent session count; no score targeting or prediction filtering',
        visualization_samples='first two generated windows per class/split, stored in dataset.npz; exact feature-row indices',
        sessions=sessions)
    return arrays, description


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sessions-per-class', type=int, default=150)
    parser.add_argument('--windows-per-session', type=int, default=40)
    parser.add_argument('--output', type=Path, default=ROOT / 'day3/out/data',
                        help='Data directory; default is the existing course path')
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    # Protect the complete data set, not just dataset.npz. Never overwrite a run.
    if out.exists() and any(out.iterdir()):
        raise SystemExit('STOP: output directory is not empty. Keep one fixed dataset for all '
                         'comparisons. Use a new directory or archive the previous day3/out run.')
    arrays, description = generate_arrays(args.seed, args.sessions_per_class,
                                          args.windows_per_session)
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'dataset.npz'
    # Unique temporary file; a concurrent process cannot have its file removed.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=out, prefix='.dataset-',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        # Link atomically without replacing another writer's dataset (same directory).
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    (out / 'labels.txt').write_text('\n'.join(LABELS) + '\n', encoding='utf-8')
    write_json(out / 'description.json', description)
    for split in SPLITS:
        print(split, arrays['x_' + split].shape,
              'sessions=', len(np.unique(arrays['session_' + split])))
    print('PASS: independent session splits, train-only normalization')
    print('Synthetic educational data only. No accuracy target is imposed.')
    print('OUTPUT:', out)


if __name__ == '__main__':
    main()

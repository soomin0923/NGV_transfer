"""Pure NumPy export checks; safe to import on WSL tools and Raspberry Pi.

Fashion export fix 2026-09-28. No TensorFlow import, model training, or transfers.
"""
from pathlib import Path
import json
import numpy as np
from extra_common.util import finite, sha256, write_json

ATOL = 1e-4
RTOL = 1e-3
VARIANTS = ('baseline', 'pruned')
PRECISIONS = ('fp32', 'dynamic', 'int8')
ONNX_PRECISIONS = ('fp32', 'int8')
SCHEMA = 'fashion_cpu_reference_v1'


def images(x):
    x = np.ascontiguousarray(finite(x, 'reference/calibration input'), dtype=np.float32)
    if x.ndim != 4 or x.shape[1:] != (28, 28, 1) or len(x) == 0:
        raise ValueError(f'Expected nonempty [N,28,28,1] input, got {x.shape}')
    if x.min() < 0 or x.max() > 1:
        raise ValueError('Fashion-MNIST input must be float32 in [0,1]. Do not normalize twice.')
    return x


def compare(actual, expected):
    a = finite(actual, 'actual output')
    b = finite(expected, 'reference output')
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 10 or len(a) == 0:
        raise ValueError(f'Output shape mismatch: actual={a.shape}, reference={b.shape}; need [N,10]')
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    close = np.isclose(a, b, atol=ATOL, rtol=RTOL)
    return dict(allclose=bool(close.all()), atol=ATOL, rtol=RTOL,
                max_abs_error=float(delta.max()), mean_abs_error=float(delta.mean()),
                mismatched_elements=int((~close).sum()),
                top1_agreement=float(np.mean(a.argmax(1) == b.argmax(1))),
                worst_index=list(np.unravel_index(int(delta.argmax()), delta.shape)))


def enforce(metrics, label):
    print(f"{label}: max_abs={metrics['max_abs_error']:.8g}, "
          f"allclose={metrics['allclose']}, top1={metrics['top1_agreement']:.6f}, "
          f"mismatched_elements={metrics['mismatched_elements']}", flush=True)
    if not metrics['allclose']:
        raise ValueError(f'{label}: FP32 parity failed at unchanged atol={ATOL}, rtol={RTOL}. '
                         'No unvalidated candidate will be published. Read the diagnostics JSON.')


def check_weights(model_weights, snapshot):
    with np.load(snapshot, allow_pickle=False) as z:
        if set(z.files) != {f'w{i}' for i in range(len(model_weights))}:
            raise ValueError('Keras/NPZ weight count mismatch. Do not mix training runs.')
        for i, w in enumerate(model_weights):
            w = finite(w, f'Keras weight {i}')
            v = finite(z[f'w{i}'], f'NPZ weight {i}')
            if w.shape != v.shape or w.dtype != v.dtype or not np.array_equal(w, v):
                raise ValueError(f'Keras and weights.npz differ at weight {i}. Do not mix training runs.')


def load_cpu_reference(out, variant):
    path = out / f'{variant}.reference_cpu.npz'
    if not path.is_file():
        raise FileNotFoundError('Missing CPU reference. Run day3_image/export_tflite.py in .venv first.')
    with np.load(path, allow_pickle=False) as z:
        ref = {k: z[k].copy() for k in z.files}
    for key, filename in [('keras_sha256', f'{variant}.keras'),
                          ('weights_sha256', f'{variant}.weights.npz'),
                          ('training_reference_sha256', f'{variant}.reference.npz')]:
        if str(ref[key].item()) != sha256(out / filename):
            raise ValueError(f'Stale CPU reference: {filename} changed. Rerun TFLite export.')
    ref['x'] = images(ref['x'])
    compare(ref['y'], ref['y'])
    compare(ref['y_zero'], ref['y_zero'])
    with np.load(out / f'{variant}.reference.npz', allow_pickle=False) as old:
        if not np.array_equal(ref['x'], old['x']):
            raise ValueError('CPU reference inputs differ from original validation samples.')
    return ref


def marker(out, kind):
    return out / f'.{kind}_export_incomplete.json'


def start_export(out, kind):
    write_json(marker(out, kind), {'status': 'incomplete', 'kind': kind})


def finish_export(out, stage, names, kind, rows):
    # All candidates have already passed. Commit report last; the marker guards partial commits.
    for name in names:
        (stage / name).replace(out / name)
    write_json(out / f'{kind}_validation.json', rows)
    marker(out, kind).unlink()


def require_validated(out, kind):
    if kind not in ('tflite', 'onnx'):
        raise ValueError(kind)
    if marker(out, kind).exists():
        raise RuntimeError(f'{kind} export is incomplete. Rerun its exporter; do not benchmark partial files.')
    precisions = PRECISIONS if kind == 'tflite' else ONNX_PRECISIONS
    expected = {(v, p) for v in VARIANTS for p in precisions}
    paths = [out / f'{v}_{p}.{kind}' for v, p in sorted(expected)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Export all {len(expected)} {kind} models before benchmarking: ' + ', '.join(missing))
    report = out / f'{kind}_validation.json'
    if not report.is_file():
        raise FileNotFoundError(f'Missing validation report: {report.name}. Rerun the updated exporter.')
    rows = json.loads(report.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError(f'Expected {len(expected)} {kind} export validation records.')
    indexed = {(r.get('variant'), r.get('precision')): r for r in rows}
    if set(indexed) != expected:
        raise ValueError('Duplicate/missing model validation records.')
    for v in VARIANTS:
        for p in precisions:
            r = indexed[v, p]
            if r.get('schema') != SCHEMA or r.get('validated') is not True:
                raise ValueError('Old or incomplete export validation report. Rerun the updated exporter.')
            if r.get('model_sha256') != sha256(out / f'{v}_{p}.{kind}'):
                raise ValueError(f'Unvalidated/replaced model: {v}_{p}.{kind}')
            # Training files do not ship to Pi. Verify them when present on WSL.
            for field, filename in [('source_keras_sha256', f'{v}.keras'),
                                    ('source_weights_sha256', f'{v}.weights.npz'),
                                    ('training_reference_sha256', f'{v}.reference.npz'),
                                    ('cpu_reference_sha256', f'{v}.reference_cpu.npz'),
                                    ('calibration_sha256', 'calibration.npz')]:
                if (out / filename).is_file() and r.get(field) != sha256(out / filename):
                    raise ValueError(f'Export report does not match {filename}. Rerun export.')
    return rows

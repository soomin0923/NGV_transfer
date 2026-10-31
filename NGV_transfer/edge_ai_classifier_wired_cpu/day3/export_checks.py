"""Portable validation gates for the nine TFLite / six ONNX CPU vibration models."""
from pathlib import Path
import numpy as np
from extra_common.util import finite, read_json, write_json, sha256

VARIANTS = ('baseline', 'unstructured', 'structured')
SCHEMA = 'vibration_cpu_reference_v2'


def compare(actual, expected, label, strict=False):
    actual, expected = finite(actual), finite(expected)
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] != 4 or not len(actual):
        raise ValueError(f'{label}: output shape mismatch; expected [N,4]')
    close = bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-3))
    result = {'max_abs_error': float(np.max(np.abs(actual - expected))),
              'allclose': close, 'top1_agreement': float(np.mean(actual.argmax(1) == expected.argmax(1)))}
    if strict and not close:
        raise ValueError(f'{label}: FP32 parity failed at atol=1e-4, rtol=1e-3. '
                         'No candidate will be published.')
    return result


def start_export(out):
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / '.export_incomplete.json', {'status': 'incomplete'})


def finish_export(out, stage, names, rows):
    for name in names:
        (stage / name).replace(out / name)
    write_json(out / 'conversion.json', rows)
    (out / '.export_incomplete.json').unlink()


def require_validated(out, backend, models=None):
    out = Path(out)
    if backend not in ('tflite', 'onnx'):
        raise ValueError(backend)
    if (out / '.export_incomplete.json').exists():
        raise RuntimeError('Vibration export incomplete. Rerun export before benchmark/transfer.')
    precisions = ('fp32', 'dynamic', 'int8') if backend == 'tflite' else ('fp32', 'int8')
    report = out / 'conversion.json'
    if not report.is_file():
        raise FileNotFoundError('Missing conversion.json. Run the CPU exporter first.')
    rows = read_json(report)
    expected = {(v, p) for v in VARIANTS for p in precisions}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError(f'Expected all {len(expected)} vibration export records; rerun the R2 exporters.')
    index = {(r.get('variant'), r.get('precision')): r for r in rows}
    if set(index) != expected:
        raise ValueError('Duplicate or missing vibration export records.')
    for (variant, precision), row in index.items():
        model = out / f'{variant}_{precision}.{backend}'
        if row.get('schema') != SCHEMA or row.get('validated') is not True or row.get('model_sha256') != sha256(model):
            raise ValueError(f'Unvalidated/replaced model: {model.name}')
        if models is not None:
            for field, suffix in [('keras_sha256', '.keras'), ('weights_sha256', '.weights.npz'),
                                  ('training_reference_sha256', '.reference.npz')]:
                path = Path(models) / (variant + suffix)
                if path.is_file() and sha256(path) != row.get(field):
                    raise ValueError(f'Stale export: {path.name} changed. Rerun exporters.')
    return rows


def source_hashes(models, variant):
    return {'keras_sha256': sha256(models / f'{variant}.keras'),
            'weights_sha256': sha256(models / f'{variant}.weights.npz'),
            'training_reference_sha256': sha256(models / f'{variant}.reference.npz')}


def load_cpu_reference(folder, models, variant):
    with np.load(folder / f'{variant}.reference_cpu.npz', allow_pickle=False) as z:
        reference = {k: z[k].copy() for k in z.files}
    for field, digest in source_hashes(models, variant).items():
        if reference[field].item() != digest:
            raise ValueError('Stale CPU reference. Rerun day3/export_tflite.py first.')
    return reference


def calibration_sample(data, count=256, seed=42):
    """Balanced, fixed-seed sampling from training features only (never val/test)."""
    x, y = data['x_train'], data['y_train']
    if count < 4:
        raise ValueError('Calibration count must be at least four')
    classes = np.arange(4)
    if not np.array_equal(np.unique(y), classes):
        raise ValueError('Calibration requires all four training classes')
    per_class = min(count // 4, min(np.sum(y == k) for k in classes))
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.choice(np.flatnonzero(y == k), per_class, replace=False) for k in classes])
    rng.shuffle(indices)
    return np.ascontiguousarray(x[indices], dtype=np.float32), indices

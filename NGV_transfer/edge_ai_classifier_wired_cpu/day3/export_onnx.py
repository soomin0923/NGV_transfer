"""Export and quantize the three vibration MLPs with CPUExecutionProvider checks."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import tempfile
import numpy as np
from extra_common.onnx_util import mlp_onnx, qdq_quantize
from extra_common.runtime import OnnxRunner
from extra_common.util import ROOT, write_json, sha256
from day3.numerics import load_weights
from day3.export_checks import (VARIANTS, SCHEMA, compare, require_validated, load_cpu_reference,
                               source_hashes, start_export, finish_export, calibration_sample)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = ROOT / 'day3/out/onnx'
    models, tflite = ROOT / 'day3/out/models', ROOT / 'day3/out/tflite'
    start_export(out)
    diagnostics = {'status': 'running', 'models': []}
    try:
        require_validated(tflite, 'tflite', models)
        with np.load(ROOT / 'day3/out/data/dataset.npz', allow_pickle=False) as z:
            cal, indices = calibration_sample(z)
            counts = np.bincount(z['y_train'][indices], minlength=4).tolist()
        with tempfile.TemporaryDirectory(prefix='.candidates_', dir=out) as tmp:
            stage, names, rows = Path(tmp), [], []
            for variant in VARIANTS:
                reference = load_cpu_reference(tflite, models, variant)
                fp32 = stage / f'{variant}_fp32.onnx'
                mlp_onnx(load_weights(models / f'{variant}.weights.npz'), fp32)
                runner = OnnxRunner(fp32)
                fp = compare(runner.predict(reference['x']), reference['y'], fp32.name, strict=True)
                del runner
                int8 = stage / f'{variant}_int8.onnx'
                qdq_quantize(fp32, int8, cal, ['MatMul'])
                runner = OnnxRunner(int8)
                quant = compare(runner.predict(reference['x']), reference['y'], int8.name)
                del runner
                for precision, path, metrics in [('fp32', fp32, fp), ('int8', int8, quant)]:
                    rows.append({'schema': SCHEMA, 'variant': variant, 'precision': precision,
                                 'validated': True, 'model_sha256': sha256(path), 'cpu_parity': metrics,
                                 'calibration_split': 'train', 'calibration_samples': len(cal),
                                 'calibration_seed': 42, 'calibration_indices': indices.tolist(),
                                 'calibration_class_counts': counts,
                                 'QuantizeBias': False, **source_hashes(models, variant)})
                    names.append(path.name)
                    print('PASS CANDIDATE:', path.name, metrics, flush=True)
            finish_export(out, stage, names, rows)
        diagnostics.update(status='passed', models=rows)
        print('PASS: all six CPU ONNX models validated and published.')
    except Exception as exc:
        diagnostics.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(out / 'diagnostics.json', diagnostics)


if __name__ == '__main__':
    main()

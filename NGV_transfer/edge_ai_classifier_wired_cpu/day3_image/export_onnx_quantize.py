"""Export ONNX FP32/INT8 using the verified CPU Keras reference."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import tempfile
import numpy as np
from extra_common.onnx_util import cnn_onnx, qdq_quantize
from extra_common.runtime import OnnxRunner
from extra_common.util import ROOT, finite, write_json, sha256
from day3.numerics import load_weights
from day3_image.export_checks import (VARIANTS, SCHEMA, images, compare, enforce,
    load_cpu_reference, start_export, finish_export, require_validated)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = ROOT / 'day3_image/artifacts'
    diagnostics = {'status': 'running', 'models': []}
    start_export(out, 'onnx')
    try:
        # CPU references are produced by the preceding, successful TFLite step.
        require_validated(out, 'tflite')
        with np.load(out / 'calibration.npz', allow_pickle=False) as z:
            cal = images(z['x'])
        refs = {v: load_cpu_reference(out, v) for v in VARIANTS}
        with tempfile.TemporaryDirectory(prefix='.onnx_candidates_', dir=out) as tmp:
            stage = Path(tmp)
            rows, names = [], []
            for variant in VARIANTS:
                ref = refs[variant]
                source = stage / f'{variant}_fp32.onnx'
                cnn_onnx(load_weights(out / f'{variant}.weights.npz'), source)
                runner = OnnxRunner(source)
                fp = compare(runner.predict(ref['x']), ref['y'])
                zp = compare(runner.predict(np.zeros((1, 28, 28, 1), np.float32)), ref['y_zero'])
                diagnostics['models'].append({'variant': variant, 'precision': 'fp32', 'cpu_parity': fp, 'zero_parity': zp})
                write_json(out / 'onnx_export_diagnostics.json', diagnostics)
                enforce(fp, f'{variant} CPU Keras -> FP32 ONNX')
                enforce(zp, f'{variant} zero-input FP32 ONNX')
                del runner
                quantized = stage / f'{variant}_int8.onnx'
                # Existing QuantizeBias=False and INT32-DQ rejection stay unchanged.
                qdq_quantize(source, quantized, cal, ['Conv', 'MatMul'])
                runner = OnnxRunner(quantized)
                qi = compare(runner.predict(ref['x']), ref['y'])
                qz = compare(runner.predict(np.zeros((1, 28, 28, 1), np.float32)), ref['y_zero'])
                del runner
                for precision, path, stats, zeros in [('fp32', source, fp, zp), ('int8', quantized, qi, qz)]:
                    row = dict(schema=SCHEMA, variant=variant, precision=precision, validated=True,
                        finite=True, QuantizeBias=False, calibration_split='train',
                        cpu_parity=stats, zero_parity=zeros, max_abs_error=stats['max_abs_error'],
                        top1_agreement=stats['top1_agreement'], model_sha256=sha256(path),
                        source_keras_sha256=sha256(out / f'{variant}.keras'),
                        source_weights_sha256=sha256(out / f'{variant}.weights.npz'),
                        training_reference_sha256=sha256(out / f'{variant}.reference.npz'),
                        cpu_reference_sha256=sha256(out / f'{variant}.reference_cpu.npz'),
                        calibration_sha256=sha256(out / 'calibration.npz'))
                    rows.append(row)
                    names.append(path.name)
                    print('PASS CANDIDATE:', path.name, stats['max_abs_error'], flush=True)
            finish_export(out, stage, names, 'onnx', rows)
        diagnostics.update(status='passed', models=rows)
        write_json(out / 'onnx_export_diagnostics.json', diagnostics)
        print('PASS: all 4 ONNX models validated and published. QuantizeBias=False.', flush=True)
    except Exception as exc:
        diagnostics.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(out / 'onnx_export_diagnostics.json', diagnostics)
        print('STOP: ONNX export incomplete. Read day3_image/artifacts/onnx_export_diagnostics.json.', flush=True)
        raise


if __name__ == '__main__':
    main()

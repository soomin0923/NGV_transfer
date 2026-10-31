"""CPU-reference FP32 validation and staged FP32/dynamic/full-INT8 export.

Original training references, learned weights and evaluation labels are never overwritten.
"""
from pathlib import Path
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Only this Python process.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import tempfile
import numpy as np
from extra_common.tf_util import tf, convert_frozen, predict_array
from extra_common.runtime import LiteRunner
from extra_common.util import ROOT, finite, write_json, sha256
from day3_image.export_checks import (VARIANTS, PRECISIONS, SCHEMA, images, compare,
    enforce, check_weights, start_export, finish_export)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = ROOT / 'day3_image/artifacts'
    out.mkdir(parents=True, exist_ok=True)
    diagnostics = {'status': 'running', 'comparison_reference': 'same saved Keras model on CPU',
                   'tensorflow': tf.__version__, 'variants': {}, 'models': []}
    start_export(out, 'tflite')
    try:
        # The subprocess is deliberately CPU-only; do not change shell/global GPU settings.
        tf.config.set_visible_devices([], 'GPU')
        tf.config.experimental.enable_tensor_float_32_execution(False)
        print('EXPORT DEVICE: CPU; original training reference is preserved.', flush=True)
        with np.load(out / 'calibration.npz', allow_pickle=False) as z:
            cal = images(z['x'])
        with tempfile.TemporaryDirectory(prefix='.tflite_candidates_', dir=out) as tmp:
            stage = Path(tmp)
            names, rows = [], []
            with tf.device('/CPU:0'):
                for variant in VARIANTS:
                    model = tf.keras.models.load_model(out / f'{variant}.keras', compile=False)
                    check_weights(model.get_weights(), out / f'{variant}.weights.npz')
                    with np.load(out / f'{variant}.reference.npz', allow_pickle=False) as z:
                        x, saved_y = images(z['x']), finite(z['y'], 'saved training reference')
                    expected = finite(predict_array(model, x, batch=64), 'CPU Keras output')
                    zero = np.zeros((1, 28, 28, 1), dtype=np.float32)
                    expected_zero = finite(predict_array(model, zero), 'CPU Keras zero output')
                    drift = compare(expected, saved_y)
                    diagnostics['variants'][variant] = {'cpu_vs_saved_training_reference': drift}
                    print(f"REFERENCE {variant}: CPU vs saved max_abs={drift['max_abs_error']:.8g}, "
                          f"allclose={drift['allclose']}, top1={drift['top1_agreement']:.6f}", flush=True)
                    if not drift['allclose']:
                        print('NOTE: saved training output and CPU output differ. '
                              'This alone does not identify TF32 or prove a bad conversion.', flush=True)
                    reference_name = f'{variant}.reference_cpu.npz'
                    np.savez_compressed(stage / reference_name, x=x, y=expected, y_zero=expected_zero,
                        keras_sha256=sha256(out / f'{variant}.keras'),
                        weights_sha256=sha256(out / f'{variant}.weights.npz'),
                        training_reference_sha256=sha256(out / f'{variant}.reference.npz'))
                    names.append(reference_name)
                    for precision in PRECISIONS:
                        name = f'{variant}_{precision}.tflite'
                        candidate = stage / name
                        candidate.write_bytes(convert_frozen(model, [None, 28, 28, 1], precision, cal))
                        runner = LiteRunner(candidate)
                        actual = finite(runner.predict(x), name)
                        metrics = compare(actual, expected)
                        zero_metrics = compare(runner.predict(zero), expected_zero)
                        row = dict(schema=SCHEMA, variant=variant, precision=precision,
                            validated=False, finite=True, max_abs_error=metrics['max_abs_error'],
                            top1_agreement=metrics['top1_agreement'], cpu_parity=metrics,
                            zero_parity=zero_metrics, cpu_vs_saved_training_reference=drift)
                        diagnostics['models'].append(row)
                        write_json(out / 'tflite_export_diagnostics.json', diagnostics)
                        if precision == 'fp32':
                            enforce(metrics, f'{variant} CPU Keras -> FP32 TFLite')
                            enforce(zero_metrics, f'{variant} zero-input FP32')
                        inp = runner.interpreter.get_input_details()[0]
                        output = runner.interpreter.get_output_details()[0]
                        required = np.dtype(np.int8 if precision == 'int8' else np.float32)
                        if np.dtype(inp['dtype']) != required or np.dtype(output['dtype']) != required:
                            raise ValueError(f'{name}: wrong input/output dtype, expected {required}')
                        if precision == 'int8' and (inp['quantization'][0] <= 0 or output['quantization'][0] <= 0):
                            raise ValueError(f'{name}: invalid INT8 scale')
                        row.update(validated=True, input_dtype=str(inp['dtype']),
                            input_quantization=list(inp['quantization']),
                            output_quantization=list(output['quantization']),
                            calibration_split='train' if precision == 'int8' else 'not_required',
                            status='PASS' if metrics['allclose'] else 'SCORE_DRIFT_EVALUATE_ACCURACY',
                            model_sha256=sha256(candidate),
                            source_keras_sha256=sha256(out / f'{variant}.keras'),
                            source_weights_sha256=sha256(out / f'{variant}.weights.npz'),
                            training_reference_sha256=sha256(out / f'{variant}.reference.npz'),
                            cpu_reference_sha256=sha256(stage / reference_name),
                            calibration_sha256=sha256(out / 'calibration.npz'))
                        rows.append(row)
                        names.append(name)
                        print(f"PASS CANDIDATE: {name}; max_abs_vs_CPU={metrics['max_abs_error']:.8g}; "
                              f"top1_agreement={metrics['top1_agreement']:.6f}", flush=True)
                        del runner
            finish_export(out, stage, names, 'tflite', rows)
        diagnostics['status'] = 'passed'
        write_json(out / 'tflite_export_diagnostics.json', diagnostics)
        print('PASS: all 6 TFLite models validated and published.', flush=True)
        print('INT8 checks validate format/finite output, not an accuracy guarantee. Run bench_cpu.py.', flush=True)
    except Exception as exc:
        diagnostics.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(out / 'tflite_export_diagnostics.json', diagnostics)
        print('STOP: export incomplete. Do not benchmark or transfer candidates. Diagnostics:',
              out / 'tflite_export_diagnostics.json', flush=True)
        raise


if __name__ == '__main__':
    main()

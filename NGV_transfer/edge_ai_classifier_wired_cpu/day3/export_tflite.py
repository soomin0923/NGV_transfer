"""Export all vibration TFLite candidates on CPU, validate, then publish."""
from pathlib import Path
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import tempfile
import numpy as np
from extra_common.tf_util import tf, convert_frozen, predict_array
from extra_common.runtime import LiteRunner
from extra_common.util import ROOT, write_json, sha256
from day3_image.export_checks import check_weights
from day3.export_checks import VARIANTS, SCHEMA, compare, source_hashes, start_export, finish_export, calibration_sample


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = ROOT / 'day3/out/tflite'
    models = ROOT / 'day3/out/models'
    start_export(out)
    diagnostics = {'status': 'running', 'models': []}
    data_path = ROOT / 'day3/out/data/dataset.npz'
    with np.load(data_path, allow_pickle=False) as data:
        calibration, calibration_indices = calibration_sample(data)
        calibration_counts = np.bincount(data['y_train'][calibration_indices], minlength=4).tolist()
    try:
        with tempfile.TemporaryDirectory(prefix='.candidates_', dir=out) as tmp:
            stage, names, rows = Path(tmp), [], []
            with tf.device('/CPU:0'):
                for variant in VARIANTS:
                    model = tf.keras.models.load_model(models / f'{variant}.keras', compile=False)
                    check_weights(model.get_weights(), models / f'{variant}.weights.npz')
                    with np.load(models / f'{variant}.reference.npz', allow_pickle=False) as z:
                        x, original_y = z['x'].copy(), z['y'].copy()
                    expected = predict_array(model, x)
                    drift = compare(expected, original_y, 'CPU/saved-training reference')
                    hashes = source_hashes(models, variant)
                    reference_name = f'{variant}.reference_cpu.npz'
                    np.savez_compressed(stage / reference_name, x=x, y=expected, **hashes)
                    names.append(reference_name)
                    for precision in ('fp32', 'dynamic', 'int8'):
                        path = stage / f'{variant}_{precision}.tflite'
                        path.write_bytes(convert_frozen(model, [None, 256], precision, calibration=calibration if precision == 'int8' else None))
                        runner = LiteRunner(path)
                        if precision == 'int8' and (runner.input['dtype'] != np.int8 or runner.output['dtype'] != np.int8):
                            raise ValueError('Full INT8 export must have int8 input and output')
                        io = {side: {'dtype': np.dtype(detail['dtype']).name,
                              'scale': float(detail['quantization'][0]), 'zero_point': int(detail['quantization'][1])}
                              for side, detail in [('input', runner.input), ('output', runner.output)]}
                        metrics = compare(runner.predict(x), expected, path.name, strict=precision == 'fp32')
                        row = {'schema': SCHEMA, 'variant': variant, 'precision': precision,
                               'validated': True, 'model_sha256': sha256(path), 'cpu_parity': metrics,
                               'cpu_vs_saved_training_reference': drift, 'io': io, **hashes}
                        if precision == 'int8':
                            row.update(calibration_split='train', calibration_seed=42,
                                calibration_samples=len(calibration), calibration_class_counts=calibration_counts,
                                calibration_indices=calibration_indices.tolist(), calibration_data_sha256=sha256(data_path))
                        rows.append(row); names.append(path.name)
                        print('PASS CANDIDATE:', path.name, metrics, flush=True)
                        del runner
            finish_export(out, stage, names, rows)
        diagnostics.update(status='passed', models=rows)
        print('PASS: all nine CPU TFLite models validated and published.')
    except Exception as exc:
        diagnostics.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(out / 'diagnostics.json', diagnostics)


if __name__ == '__main__':
    main()

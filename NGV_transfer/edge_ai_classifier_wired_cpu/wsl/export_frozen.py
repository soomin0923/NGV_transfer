"""Export CPU Keras inference to FP32, dynamic, or strict full-integer INT8 TFLite.

Calibration uses only training images. Validation agreement measures conversion
change, not accuracy: use evaluate_tflite.py on the untouched test split.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import json
import shutil
from collections import Counter
import numpy as np
from PIL import Image
from extra_common.util import finite, labels, write_json, sha256

from extra_common.calibration import image_files, load_image, load_training_calibration


def tensor_info(detail):
    return {'dtype': np.dtype(detail['dtype']).name, 'shape': [int(v) for v in detail['shape']],
            'scale': float(detail['quantization'][0]), 'zero_point': int(detail['quantization'][1])}


def verify_full_int8(interpreter):
    ins, outs = interpreter.get_input_details(), interpreter.get_output_details()
    for d in ins + outs:
        if np.dtype(d['dtype']) != np.dtype('int8') or d['quantization'][0] <= 0:
            raise ValueError('Full INT8 requires int8 input/output and positive quantization scales. No float fallback is allowed.')
    floats = [d['name'] for d in interpreter.get_tensor_details() if np.issubdtype(d['dtype'], np.floating)]
    if floats:
        raise ValueError('Full INT8 contains floating tensors: ' + ', '.join(floats[:5]))
    return {'input': tensor_info(ins[0]), 'output': tensor_info(outs[0]), 'floating_tensor_count': len(floats),
            'supported_ops': 'TFLITE_BUILTINS_INT8', 'note': 'Bias/shape tensors may be INT32; full integer does not mean every tensor is 8-bit.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='artifacts/best_model.keras')
    p.add_argument('--labels', default='artifacts/labels.txt')
    p.add_argument('--output', default='artifacts')
    p.add_argument('--sample-dir', default='dataset_split/val')
    p.add_argument('--max-images', type=int, default=400)
    p.add_argument('--calibration-dir', default='dataset_split/train')
    p.add_argument('--calibration-images', type=int, default=128)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--float32-only', action='store_true')
    g.add_argument('--diagnose-dynamic', action='store_true')
    g.add_argument('--int8', action='store_true')
    a = p.parse_args()
    if a.max_images < 1:
        p.error('--max-images must be positive')
    from extra_common.tf_util import tf, convert_frozen, predict_array
    from extra_common.runtime import LiteRunner
    model = tf.keras.models.load_model(a.model, compile=False)
    for i, weight in enumerate(model.get_weights()):
        finite(weight, f'Keras weight {i}')
    names = labels(a.labels, int(model.output_shape[-1]))
    h, w = map(int, model.input_shape[1:3])
    paths = sorted(p for c in names for p in image_files(Path(a.sample_dir) / c))[:a.max_images]
    if not paths:
        raise ValueError('No real validation images. Collect, split and train first.')
    real_x = np.stack([load_image(f, h, w) for f in paths])
    x = np.concatenate([np.zeros((1, h, w, 3), np.float32), real_x])
    ref = finite(predict_array(model, x, 16), 'Keras reference')
    mode = 'int8' if a.int8 else ('dynamic' if a.diagnose_dynamic else 'fp32')
    subdir = {'fp32': '', 'dynamic': 'dynamic_diagnostic', 'int8': 'int8'}[mode]
    out = Path(a.output) / subdir
    out.mkdir(parents=True, exist_ok=True)
    name = {'fp32': 'model_float32_frozen.tflite', 'dynamic': 'model_dynamic_frozen.tflite',
            'int8': 'model_int8_frozen.tflite'}[mode]
    calibration = None
    calibration_report = None
    if a.int8:
        calibration, calibration_report = load_training_calibration(a.calibration_dir, (h,w), a.calibration_images, names=names)
    dst = out / name
    tmp = dst.with_suffix('.tflite.tmp')
    try:
        tmp.write_bytes(convert_frozen(model, [None, h, w, 3], mode, calibration=calibration))
        runner = LiteRunner(tmp, 4)
        integer_info = verify_full_int8(runner.interpreter) if a.int8 else None
        pred = finite(np.concatenate([runner.predict(v[None]) for v in x]), 'LiteRT outputs')
        if pred.shape != ref.shape:
            raise ValueError('Output shape differs from Keras')
        probability_tolerance = max(.01, len(names) / 512 + .001) if a.int8 else .01
        if not np.allclose(pred.sum(1), 1, atol=probability_tolerance):
            raise ValueError('Invalid probability sums')
        max_abs = float(np.max(np.abs(pred - ref)))
        numerical_match = bool(np.allclose(pred, ref, atol=2e-4, rtol=2e-3))
        if mode == 'fp32' and not numerical_match:
            raise ValueError(f'FP32 numerical validation failed: max_abs={max_abs}. Candidate not promoted.')
        status = 'PASS' if numerical_match else 'DIAGNOSTIC_SCORE_DRIFT'
        report = {'schema': 'day2_export_v2', 'status': status, 'mode': mode,
                  'source_sha256': sha256(a.model), 'model_sha256': sha256(tmp), 'finite': True,
                  'samples_including_zero': len(x), 'validation_real_samples': len(real_x),
                  'max_abs_error': max_abs, 'mean_abs_error': float(np.mean(np.abs(pred-ref))),
                  'keras_top1_agreement': float(np.mean(pred.argmax(1) == ref.argmax(1))),
                  'real_images_keras_top1_agreement': float(np.mean(pred[1:].argmax(1) == ref[1:].argmax(1))),
                  'validation_data': str(Path(a.sample_dir).resolve()),
                  'input': tensor_info(runner.input), 'output': tensor_info(runner.output),
                  'not_accuracy': 'Agreement is with Keras predictions, not ground truth. Evaluate untouched test images for accuracy.',
                  'integer_validation': integer_info, 'calibration': calibration_report,
                  'execution_path': runner.execution_path, 'delegate_fallback_reason': runner.delegate_fallback_reason}
        tmp.replace(dst)
        if Path(a.labels).resolve() != (out / 'labels.txt').resolve():
            shutil.copy2(a.labels, out / 'labels.txt')
        if calibration_report:
            write_json(out / 'calibration_manifest.json', calibration_report)
        write_json(out / 'frozen_validation.json', report)
        print(f'{status}: {dst}; Keras agreement (real validation)={report["real_images_keras_top1_agreement"]:.4f}; max_abs={max_abs:.6g}')
        if a.int8:
            print('PASS: strict integer operators, int8 I/O with scale/zero-point verified. Evaluate test accuracy before deployment.')
    finally:
        if tmp.exists():
            tmp.unlink()


if __name__ == '__main__':
    main()

"""Measure true held-out image accuracy and CPU latency for one TFLite model."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import time
import numpy as np
from PIL import Image
from extra_common.runtime import LiteRunner
from extra_common.util import labels, write_json, save_csv, sha256, finite, metadata
from wsl.export_frozen import image_files, load_image, tensor_info


def classification_metrics(cm, names):
    cm = np.asarray(cm, dtype=np.int64)
    support = cm.sum(1)
    predicted = cm.sum(0)
    tp = np.diag(cm)
    precision = np.divide(tp, predicted, out=np.zeros(len(names), float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(names), float), where=support > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(len(names), float), where=(precision+recall)>0)
    return {'accuracy': float(tp.sum()/cm.sum()), 'macro_f1': float(f1.mean()),
            'per_class': [{'label': label, 'support': int(support[i]), 'predicted': int(predicted[i]),
                           'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])}
                          for i, label in enumerate(names)],
            'zero_division': 'Undefined precision/recall/F1 is reported as 0.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--precision', choices=['fp32', 'dynamic', 'int8'], help='Usually inferred from export metadata')
    a = p.parse_args()
    if a.warmup < 0:
        p.error('--warmup must be nonnegative')
    runner = LiteRunner(a.model, a.threads)
    names = labels(a.labels)
    shape = runner.input['shape']
    h, w = map(int, shape[1:3])
    export_record = Path(a.model).parent / 'frozen_validation.json'
    export = json.loads(export_record.read_text()) if export_record.exists() else {}
    if export and export['model_sha256'] != sha256(a.model):
        raise ValueError('Model changed since conversion validation. Re-export before evaluating.')
    precision = a.precision or export.get('mode') or ('int8' if runner.input['dtype'] == np.int8 else 'unknown_float')
    rows = []
    cm = np.zeros((len(names), len(names)), dtype=np.int64)
    files = [(actual, f) for actual, name in enumerate(names) for f in image_files(Path(a.data) / name)]
    if any(not image_files(Path(a.data)/name) for name in names):
        raise ValueError('Each configured label must contain evaluation images.')
    first = load_image(files[0][1], h, w)[None]
    for _ in range(a.warmup):
        finite(runner.predict(first), 'warmup')
    digest = hashlib.sha256()
    elapsed, invoke_elapsed = [], []
    for actual, f in files:
        x = load_image(f, h, w)[None]
        image_hash = sha256(f)
        relative = f.resolve().relative_to(Path(a.data).resolve()).as_posix()
        digest.update((relative + '\0' + names[actual] + '\0' + image_hash + '\n').encode())
        start = time.perf_counter_ns()
        prob = finite(runner.predict(x)[0], 'evaluation probabilities')
        ms = (time.perf_counter_ns() - start) / 1e6
        if len(prob) != len(names):
            raise ValueError('Labels/model mismatch')
        pred = int(prob.argmax())
        cm[actual, pred] += 1
        elapsed.append(ms)
        invoke_elapsed.append(runner.last_invoke_ms)
        rows.append({'file': str(f.resolve()), 'relative_file': relative, 'sha256': image_hash,
                     'truth_index': actual, 'truth': names[actual], 'prediction_index': pred,
                     'prediction': names[pred], 'correct': int(actual == pred),
                     'confidence': float(prob[pred]), 'ms': ms, 'invoke_ms': runner.last_invoke_ms})
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out/'predictions.csv', rows)
    save_csv(out/'confusion_matrix.csv', [{'truth': c, **{n: int(cm[i,j]) for j,n in enumerate(names)}} for i,c in enumerate(names)])
    scores = classification_metrics(cm, names)
    save_csv(out/'per_class.csv', scores['per_class'])
    report = {'schema': 'day2_evaluation_v2', **metadata(), **scores,
              'samples': len(rows), 'labels': names, 'confusion_matrix': cm.tolist(),
              'model': str(Path(a.model).resolve()), 'model_sha256': sha256(a.model),
              'bytes': Path(a.model).stat().st_size, 'data': str(Path(a.data).resolve()),
              'data_sha256': digest.hexdigest(), 'split': Path(a.data).name,
              'backend': 'litert', 'device': 'CPU', 'precision': precision, 'threads': a.threads,
              'execution_path': runner.execution_path, 'delegate_fallback_reason': runner.delegate_fallback_reason,
              'batch': 1, 'warmup': a.warmup, 'timed_samples': len(rows),
              'timing_scope': 'prepared float32 input -> quantization (if any) -> CPU inference -> float32 probabilities; excludes file load/resize and camera/display',
              'latency_method': 'one inference per held-out image after warmup',
              'median_ms': float(np.median(elapsed)), 'p95_ms': float(np.percentile(elapsed,95)),
              'mean_ms': float(np.mean(elapsed)), 'invoke_median_ms': float(np.median(invoke_elapsed)),
              'input': tensor_info(runner.input), 'output': tensor_info(runner.output)}
    write_json(out/'metrics.json', report)
    print(f'{precision}: n={len(rows)}, accuracy={scores["accuracy"]:.4f}, macro-F1={scores["macro_f1"]:.4f}, median={report["median_ms"]:.3f} ms, p95={report["p95_ms"]:.3f} ms')
    print('Saved:', out/'metrics.json')


if __name__ == '__main__':
    main()

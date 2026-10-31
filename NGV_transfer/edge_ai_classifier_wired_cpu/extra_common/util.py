"""Shared I/O, reproducibility and benchmark metadata. No network-transfer wrappers."""
from __future__ import annotations
import csv, hashlib, html, json, os, platform, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')
    tmp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def finite(x, name='array'):
    a = np.asarray(x)
    if not np.isfinite(a).all(): raise ValueError(f'{name}: NaN/Inf detected. STOP.')
    return a

def labels(path, count=None, allow_duplicates=False):
    out = [s.strip() for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not out or (not allow_duplicates and len(out) != len(set(out))): raise ValueError(f'Empty/duplicate labels: {path}')
    if count is not None and len(out) != count: raise ValueError(f'Label count {len(out)} != model outputs {count}')
    return out

def metadata():
    return dict(utc=datetime.now(timezone.utc).isoformat(), hostname=platform.node(),
                machine=platform.machine(), platform=platform.platform(), python=platform.python_version())

def save_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: raise ValueError('No rows to save')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def measure(runner, x, y, *, model, backend, tag, batch=1, threads=4,
            warmup=20, repeats=100, variant='', precision=''):
    """Latency: prepared float32 batch -> runtime -> NumPy probabilities."""
    if batch < 1 or repeats < 1 or warmup < 0: raise ValueError('Invalid benchmark counts')
    if len(x) != len(y) or len(x) == 0: raise ValueError('Empty/mismatched evaluation data')
    xb=np.ascontiguousarray(x[:min(batch,len(x))], dtype=np.float32)
    for _ in range(warmup): finite(runner.predict(xb), 'warmup output')
    elapsed=[]
    for _ in range(repeats):
        t=time.perf_counter_ns(); out=runner.predict(xb); dt=(time.perf_counter_ns()-t)/1e6
        finite(out, 'timed output'); elapsed.append(dt)
    correct=0; confusion=None
    for start in range(0,len(x),batch):
        pred=runner.predict(np.ascontiguousarray(x[start:start+batch],dtype=np.float32))
        finite(pred,'evaluation output')
        if pred.ndim != 2: raise ValueError('Classifier output must be [batch, classes]')
        if confusion is None: confusion=np.zeros((pred.shape[1],pred.shape[1]),dtype=np.int64)
        truth=np.asarray(y[start:start+batch],dtype=np.int64)
        guessed=np.argmax(pred,axis=1)
        if len(truth)!=len(guessed) or np.any(truth<0) or np.any(truth>=len(confusion)):
            raise ValueError('Invalid ground-truth labels/output size')
        np.add.at(confusion,(truth,guessed),1)
        correct+=int(np.sum(guessed==truth))
    tp=np.diag(confusion).astype(float)
    recall=np.divide(tp,confusion.sum(1),out=np.zeros_like(tp),where=confusion.sum(1)>0)
    precision_scores=np.divide(tp,confusion.sum(0),out=np.zeros_like(tp),where=confusion.sum(0)>0)
    f1=np.divide(2*recall*precision_scores,recall+precision_scores,out=np.zeros_like(tp),where=(recall+precision_scores)>0)
    return dict(schema='edge_benchmark_v1', **metadata(), backend=backend, tag=tag,
                variant=variant, precision=precision, model=Path(model).name,
                execution_path=getattr(runner,'execution_path','onnx_cpu' if backend=='onnx' else 'cpu'),
                model_sha256=sha256(model), bytes=Path(model).stat().st_size,
                input_data_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()+np.ascontiguousarray(y).tobytes()).hexdigest(),
                batch=len(xb), threads=threads,
                samples=len(x), accuracy=correct/len(x), macro_f1=float(f1.mean()),
                confusion_matrix=confusion.tolist(), class_indices=list(range(len(confusion))),
                per_class_precision=precision_scores.tolist(), per_class_recall=recall.tolist(),
                per_class_f1=f1.tolist(), warmup=warmup,repeats=repeats,
                timing_scope='prepared_input_to_numpy_output; CPU only',
                mean_ms=float(np.mean(elapsed)), median_ms=float(np.median(elapsed)),
                p95_ms=float(np.percentile(elapsed,95)),
                samples_per_second=float(len(xb)*1000/np.mean(elapsed)))

def write_report(rows, target):
    target=Path(target); target.mkdir(parents=True,exist_ok=True)
    if not rows: raise ValueError('No benchmark records. Run the benchmarks first.')
    save_csv(target/'results.csv', rows); write_json(target/'results_full.json',rows)
    keys=['tag','backend','variant','precision','batch','threads','samples','accuracy','median_ms','p95_ms','samples_per_second','bytes']
    def fmt(k,v):
        if v is None:return '-'
        if k=='accuracy':return f'{100*v:.2f}%'
        if isinstance(v,float):return f'{v:.4f}'
        return str(v)
    body=''.join('<tr>'+''.join('<td>'+html.escape(fmt(k,r.get(k)))+'</td>' for k in keys)+'</tr>' for r in rows)
    header=''.join('<th>'+html.escape(k)+'</th>' for k in keys)
    text='<!doctype html><meta charset="utf-8"><title>Edge AI results</title><style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}th{background:#eee}</style><h1>Edge AI — measured results</h1><p>Recorded CPU results. Compare only matching evaluation data, device, threads and batch settings. Input preparation and camera/display latency are excluded.</p><div style="overflow:auto"><table><tr>'+header+'</tr>'+body+'</table></div>'
    (target/'results.html').write_text(text,encoding='utf-8')
    md='| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'
    md+='\n'.join('| '+' | '.join(fmt(k,r.get(k)) for k in keys)+' |' for r in rows)
    (target/'results.md').write_text(md+'\n',encoding='utf-8')
    print('REPORT:',target/'results.html')

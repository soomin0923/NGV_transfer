from pathlib import Path
import argparse,time
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=32);p.add_argument('--interval',type=float,default=.5);p.add_argument('--precision',choices=['fp32','dynamic','int8'],default='fp32');p.add_argument('--variant',choices=['baseline','unstructured','structured'],default='structured');p.add_argument('--threads',type=int,default=4);a=p.parse_args()
import numpy as np
from day3.numerics import LABELS,signal,features
from extra_common.runtime import LiteRunner
from day3.export_checks import require_validated
base=Path(__file__).resolve().parent;require_validated(base/'models','tflite');r=LiteRunner(base/f'models/{a.variant}_{a.precision}.tflite',a.threads);z=np.load(base/'data/dataset.npz',allow_pickle=False);rng=np.random.default_rng(2026)
print('SYNTHETIC SIGNAL DEMO — not a real sensor or validated fault detector')
for i in range(a.steps):
    kind=i%4;x=((features(signal(kind,rng))-z['mean'])/z['std']).astype(np.float32)[None]
    pred=r.predict(x)[0];print(i+1,'truth=',LABELS[kind],'prediction=',LABELS[int(pred.argmax())],'score=',float(pred.max()),flush=True);time.sleep(a.interval)

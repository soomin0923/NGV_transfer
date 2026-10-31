from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=4);p.add_argument('--batch',type=int,default=1);a=p.parse_args()
import numpy as np
from extra_common.runtime import LiteRunner
from day3.export_checks import require_validated
from extra_common.util import measure,write_json,sha256
base=Path(__file__).resolve().parent;require_validated(base/'models','tflite');z=np.load(base/'data/dataset.npz',allow_pickle=False);rows=[]
for variant in ['baseline','unstructured','structured']:
    for precision in ['fp32','dynamic','int8']:
        path=base/f'models/{variant}_{precision}.tflite';r=LiteRunner(path,a.threads)
        row=measure(r,z['x_test'],z['y_test'],model=path,backend='tflite',tag='pi',batch=a.batch,threads=a.threads,variant=variant,precision=precision);row['evaluation_sha256']=sha256(base/'data/dataset.npz');rows.append(row);print(variant,precision,row['accuracy'],row['median_ms'])
write_json(base/f'results/pi_tflite_batch{a.batch}.json',rows)

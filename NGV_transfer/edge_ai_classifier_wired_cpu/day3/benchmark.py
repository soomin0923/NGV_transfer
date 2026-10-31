from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['tflite','onnx'],required=True);p.add_argument('--batch',type=int,default=1);p.add_argument('--threads',type=int,default=4);p.add_argument('--repeats',type=int,default=100);a=p.parse_args()
import numpy as np
from extra_common.util import ROOT,measure,write_json,sha256
from extra_common.runtime import LiteRunner,OnnxRunner
from day3.export_checks import require_validated
require_validated(ROOT/f'day3/out/{a.backend}',a.backend,ROOT/'day3/out/models')
z=np.load(ROOT/'day3/out/data/dataset.npz',allow_pickle=False);variants=['baseline','unstructured','structured']
precisions=['fp32','dynamic','int8'] if a.backend=='tflite' else ['fp32','int8']
results=[]
for variant in variants:
    for precision in precisions:
        if a.backend=='tflite':path=ROOT/f'day3/out/tflite/{variant}_{precision}.tflite';r=LiteRunner(path,a.threads)
        elif a.backend=='onnx':path=ROOT/f'day3/out/onnx/{variant}_{precision}.onnx';r=OnnxRunner(path,a.threads)
        try:row=measure(r,z['x_test'],z['y_test'],model=path,backend=a.backend,tag='wsl',batch=a.batch,threads=a.threads,repeats=a.repeats,variant=variant,precision=precision)
        finally:
            if hasattr(r,'close'):r.close()
        row['evaluation_sha256']=sha256(ROOT/'day3/out/data/dataset.npz');results.append(row);print(variant,precision,row['accuracy'],row['median_ms'])
write_json(ROOT/f'day3/out/results/wsl_{a.backend}_batch{a.batch}.json',results)

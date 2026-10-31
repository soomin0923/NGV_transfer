from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['litert','onnx'],required=True);p.add_argument('--tag',choices=['wsl','pi'],required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--batch',type=int,default=1);p.add_argument('--repeats',type=int,default=100);a=p.parse_args()
import numpy as np
from extra_common.util import ROOT,measure,write_json,save_csv,sha256
from extra_common.runtime import LiteRunner,OnnxRunner
from day3_image.export_checks import require_validated, PRECISIONS, ONNX_PRECISIONS
out=ROOT/'day3_image/artifacts'
# Refuse partial exports before measuring even the first model.
require_validated(out, 'tflite' if a.backend=='litert' else 'onnx')
z=np.load(out/'evaluation.npz',allow_pickle=False);rows=[]
for variant in ['baseline','pruned']:
    for precision in (PRECISIONS if a.backend=='litert' else ONNX_PRECISIONS):
        path=out/f'{variant}_{precision}.{ "tflite" if a.backend=="litert" else "onnx"}'
        r=LiteRunner(path,a.threads) if a.backend=='litert' else OnnxRunner(path,a.threads)
        row=measure(r,z['x'],z['y'],model=path,backend=a.backend,tag=a.tag,batch=a.batch,threads=a.threads,repeats=a.repeats,variant=variant,precision=precision);row['evaluation_sha256']=sha256(out/'evaluation.npz');rows.append(row);print(variant,precision,row['accuracy'],row['median_ms'])
result=ROOT/'day3_image/results';write_json(result/f'{a.tag}_{a.backend}.json',rows);save_csv(result/f'{a.tag}_{a.backend}.csv',rows)

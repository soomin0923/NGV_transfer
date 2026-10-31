from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse,zipfile
p=argparse.ArgumentParser();p.parse_args()
from extra_common.util import ROOT
from day3_image.export_checks import require_validated
require_validated(ROOT/'day3_image/artifacts', 'tflite')
files=[ROOT/'day3_image/bench_cpu.py',ROOT/'day3_image/review_pi.py',ROOT/'day3_image/__init__.py',ROOT/'day3_image/export_checks.py']
files += [ROOT/'extra_common'/name for name in ['__init__.py','util.py','runtime.py','webui.py']]
files += [ROOT/'day3_image/artifacts'/name for name in ['evaluation.npz','labels.txt','tflite_validation.json','baseline_fp32.tflite','baseline_dynamic.tflite','baseline_int8.tflite','pruned_fp32.tflite','pruned_dynamic.tflite','pruned_int8.tflite']]
for f in files:
    if not f.is_file():raise FileNotFoundError(f)
path=ROOT/'day3_image/pi_bundle.zip'
with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
    for f in files:z.write(f,str(f.relative_to(ROOT)))
print('PASS:',path,'(CPU models and fixed evaluation data only)')

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse,tarfile,io
p=argparse.ArgumentParser();p.parse_args()
from extra_common.util import ROOT
from day3.export_checks import require_validated
require_validated(ROOT/'day3/out/tflite','tflite',ROOT/'day3/out/models')
out=ROOT/'day3/out/day3_pi_bundle.tar.gz'
# Include a standalone directory with only portable Python, CPU models and evaluation data.
files={}
for src in (ROOT/'extra_common').glob('*.py'):
    if src.name in ['__init__.py','util.py','runtime.py']:files['day3_pi/extra_common/'+src.name]=src
for src in [ROOT/'day3/__init__.py',ROOT/'day3/numerics.py',ROOT/'day3/export_checks.py']:files['day3_pi/day3/'+src.name]=src
files['day3_pi/models/conversion.json']=ROOT/'day3/out/tflite/conversion.json'
files['day3_pi/data/dataset.npz']=ROOT/'day3/out/data/dataset.npz'
for variant in ['baseline','unstructured','structured']:
    for precision in ['fp32','dynamic','int8']:files[f'day3_pi/models/{variant}_{precision}.tflite']=ROOT/f'day3/out/tflite/{variant}_{precision}.tflite'
files['day3_pi/pi_benchmark.py']=ROOT/'day3/pi_benchmark.py';files['day3_pi/pi_demo.py']=ROOT/'day3/pi_demo.py'
for name,src in files.items():
    if not src.is_file():raise FileNotFoundError(src)
with tarfile.open(out,'w:gz') as tar:
    for name,src in files.items():tar.add(src,arcname=name)
print('PASS:',out)

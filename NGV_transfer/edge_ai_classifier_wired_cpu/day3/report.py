from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
p=argparse.ArgumentParser();p.add_argument('--extra',default='day3/out/pi_results');a=p.parse_args()
from extra_common.util import ROOT,read_json,write_report
rows=[]
for folder in [ROOT/'day3/out/results',Path(a.extra)]:
    for f in sorted(folder.glob('*.json')):
        doc=read_json(f)
        for row in doc if isinstance(doc,list) else [doc]:
            if row.get('schema')=='edge_benchmark_v1':rows.append(row)
write_report(rows,ROOT/'day3/out/report')

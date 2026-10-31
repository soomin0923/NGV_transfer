"""Table of measured benchmarks that match the current models/evaluation.

For curves, confusion matrices and charts run analyze_results.py in .venv.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from extra_common.util import ROOT,read_json,write_report,sha256,write_json
from day3_image.export_checks import require_validated


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    out=ROOT/'day3_image/artifacts';evaluation=sha256(out/'evaluation.npz')
    rows=[];excluded=[];validated=set()
    for path in sorted((ROOT/'day3_image/results').glob('*.json')):
        doc=read_json(path)
        if not isinstance(doc,list):continue
        for row in doc:
            if row.get('schema')!='edge_benchmark_v1':continue
            model=out/str(row.get('model',''));kind='onnx' if row.get('backend')=='onnx' else 'tflite'
            if row.get('evaluation_sha256')!=evaluation or not model.is_file() or row.get('model_sha256')!=sha256(model):
                excluded.append({'file':path.name,'model':row.get('model'),'reason':'current model/evaluation checksum mismatch or missing model'});continue
            if kind not in validated:require_validated(out,kind);validated.add(kind)
            rows.append(row)
    target=ROOT/'day3_image/report'
    write_json(target/'excluded_records.json',excluded)
    if excluded:print(f'Excluded {len(excluded)} stale/missing-model records; see excluded_records.json')
    write_report(rows,target)
    print('For plots: source day3/env_day2.sh && python day3_image/analyze_results.py')


if __name__=='__main__':main()

"""CPU timing with a fixed random prepared input; never an accuracy test."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import time
import numpy as np
from extra_common.runtime import LiteRunner
from extra_common.util import finite, write_json, metadata, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--output',help='Optional JSON record for the offline analysis report')
    a=p.parse_args()
    if a.repeats<1 or a.warmup<0:p.error('repeats must be positive; warmup must be nonnegative')
    runner=LiteRunner(a.model,a.threads)
    shape=tuple(int(n) for n in runner.input['shape'])
    finite(runner.predict(np.zeros(shape,np.float32)),'zero-input output')
    x=np.random.default_rng(42).uniform(0,255,shape).astype(np.float32)
    for _ in range(a.warmup):finite(runner.predict(x))
    elapsed=[]
    for _ in range(a.repeats):
        start=time.perf_counter_ns();finite(runner.predict(x));elapsed.append((time.perf_counter_ns()-start)/1e6)
    model=Path(a.model)
    precision='int8' if runner.input['dtype']==np.int8 else ('dynamic' if 'dynamic' in model.name else 'fp32')
    report={'schema':'day2_random_benchmark_v2',**metadata(),'model':model.name,'model_sha256':sha256(model),
            'execution_path':runner.execution_path,'delegate_fallback_reason':runner.delegate_fallback_reason,
            'precision':precision,'bytes':model.stat().st_size,'threads':a.threads,'repeats':a.repeats,'warmup':a.warmup,
            'input_shape':list(shape),'input_seed':42,'input_distribution':'uniform float32 [0,255)',
            'timing_scope':'prepared input -> quantization -> CPU inference -> float32 NumPy output; no camera/display',
            'median_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95)),
            'mean_ms':float(np.mean(elapsed)),'not_accuracy':'Random input has no ground truth; this is a timing/sanity test.'}
    if a.output:write_json(a.output,report)
    print('PASS: finite outputs; prepared-input prediction median_ms=',report['median_ms'],'p95_ms=',report['p95_ms'])
    print(report['not_accuracy'])
    if a.output:print('Saved:',a.output)


if __name__=='__main__':main()

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse,threading,io,time
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8080);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
import numpy as np
from PIL import Image
from extra_common.util import ROOT,labels
from extra_common.runtime import LiteRunner
from day3_image.export_checks import require_validated
from extra_common.webui import serve,page_html
class Review:
    mode='fashion-review'
    def __init__(self):
        self.lock=threading.RLock();self.index=0;self.rng=np.random.default_rng();folder=ROOT/'day3_image/artifacts';require_validated(folder,'tflite')
        z=np.load(folder/'evaluation.npz',allow_pickle=False);self.x=z['x'];self.y=z['y'];self.names=labels(folder/'labels.txt')
        self.runners={f'{v}_{p}':LiteRunner(folder/f'{v}_{p}.tflite',a.threads) for v in ['baseline','pruned'] for p in ['fp32','dynamic','int8']};self.refresh()
    def refresh(self):
        self.pred={}
        for name,r in self.runners.items():
            t=time.perf_counter();p=r.predict(self.x[self.index:self.index+1])[0];ms=(time.perf_counter()-t)*1000
            self.pred[name]={'prediction':self.names[int(p.argmax())],'confidence':float(p.max()),'ms':ms}
        image=Image.fromarray(np.uint8(np.clip(self.x[self.index,:,:,0]*255,0,255))).resize((336,336),Image.Resampling.NEAREST)
        buf=io.BytesIO();image.save(buf,format='JPEG');self.jpg=buf.getvalue()
    def page(self,token):
        return page_html('Fashion-MNIST — 저장된 평가 이미지',"<p>웹캠이 아닙니다. 같은 평가 이미지를 여섯 모델에 입력합니다. baseline/pruned × FP32/dynamic/INT8이며 단일 이미지 시간은 반복 벤치마크와 다릅니다.</p><button onclick=\"act('/move',{step:-1})\">이전</button><button onclick=\"act('/move',{step:1})\">다음</button><button onclick=\"act('/random')\">무작위</button>",token)
    def image(self):
        with self.lock:return self.jpg
    def state(self):
        with self.lock:return {'index':self.index,'total':len(self.x),'truth':self.names[int(self.y[self.index])],'predictions':self.pred,'timing_note':'single sample, not a benchmark average'}
    def action(self,path,data):
        with self.lock:
            if path=='/move':self.index=(self.index+int(data.get('step',1)))%len(self.x)
            elif path=='/random':self.index=int(self.rng.integers(len(self.x)))
            else:raise ValueError('Unknown action')
            self.refresh()
        return {'ok':True}
    def close(self):pass
serve(Review(),a.port)

from __future__ import annotations
import csv,html,threading,time,uuid
from pathlib import Path
import numpy as np
from extra_common.camera import Camera,roi_bounds,crop_roi,model_input
from extra_common.runtime import LiteRunner
from extra_common.util import ROOT,labels,finite,sha256,metadata
from extra_common.webui import page_html
from extra_common.course import load_config

def _names(values, label):
    import re
    values=list(values)
    if not values or len(values)!=len(set(values)) or any(not isinstance(v,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',v) for v in values):
        raise ValueError(f'{label}: use distinct ASCII letters/digits/underscore/hyphen names')
    if 'out_of_scope' in values:raise ValueError('out_of_scope is reserved for comparison samples')
    return values

class LiveApp:
    def __init__(self,camera_options,mode='collect',model=None,label_file=None,threads=None,threshold=.65,target=None,interval=.5,resize_mode='stretch',imagenet_model=None,imagenet_labels=None,output=None,config=None):
        course=load_config() if config is None else config
        self.classes=_names(course['labels'],'labels');self.sessions=_names(course['sessions'],'sessions')
        threads=course['threads'] if threads is None else threads
        target=course['photos_per_session'] if target is None else target
        self.truths=self.classes+['out_of_scope']
        self.mode=mode;self.lock=threading.RLock();self.stop=threading.Event()
        self.threshold=threshold;self.target=target;self.interval=interval;self.resize_mode=resize_mode
        if not 0<=threshold<=1 or target<1 or interval<=0:raise ValueError('Invalid threshold/target/interval')
        self.root=Path(output) if output else ROOT/'dataset_raw'
        self.cls=self.classes[0];self.session=self.sessions[0];self.collect=False;self.last_saved=0;self.error=''
        self.runners={};self.names={};self.model_info={};self.threads=threads
        for key,path in [('custom',model),('imagenet',imagenet_model)]:
            if path:self.model_info[key]={'model':Path(path).name,'model_sha256':sha256(path)}
        if model:
            if not label_file:raise ValueError('A custom model requires its matching labels.txt')
            self.names['custom']=labels(label_file)
            if self.names['custom']!=self.classes:
                raise ValueError('Model labels/order do not match course_config.json. Use the configuration used for training.')
            self.runners['custom']=LiteRunner(model,threads)
        if imagenet_model:
            self.runners['imagenet']=LiteRunner(imagenet_model,threads);self.names['imagenet']=labels(imagenet_labels,1000,allow_duplicates=True)
        if mode=='infer' and 'custom' not in self.runners:raise ValueError('Infer requires --model and --labels')
        self.cam=Camera(camera_options);self.jpg=b'';self.last={};self.raw=None;self.sequence=0;self.last_recorded_sequence=-1
        self.worker=threading.Thread(target=self.loop,daemon=True);self.worker.start()
    def count(self):return len(list((self.root/self.cls/self.session).glob('*.jpg')))
    def loop(self):
        import cv2
        try:
            while not self.stop.is_set():
                frame=self.cam.read();clean=crop_roi(frame);preview=frame.copy();pred={}
                for key,runner in self.runners.items():
                    shape=runner.interpreter.get_input_details()[0]['shape']
                    x=model_input(clean,int(shape[1]),int(shape[2]),self.resize_mode)
                    t=time.perf_counter();prob=finite(runner.predict(x)[0]);elapsed=(time.perf_counter()-t)*1000
                    if len(prob)!=len(self.names[key]):raise ValueError('Model/labels output count mismatch')
                    top=np.argsort(prob)[::-1][:5]
                    pred[key]={'top1':self.names[key][int(top[0])],'confidence':float(prob[top[0]]),
                               'uncertain':bool(prob[top[0]]<self.threshold),'inference_ms':elapsed,
                               'top5':[{'label':self.names[key][int(i)],'probability':float(prob[i])} for i in top]}
                with self.lock:
                    self.raw=clean;self.last=pred;self.sequence+=1
                    if self.mode=='collect' and self.collect and time.monotonic()-self.last_saved>=self.interval:
                        if self.count()>=self.target:self.collect=False
                        else:
                            dst=self.root/self.cls/self.session;dst.mkdir(parents=True,exist_ok=True)
                            name=dst/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:12]+'.jpg')
                            if not cv2.imwrite(str(name),clean,[cv2.IMWRITE_JPEG_QUALITY,95]):raise RuntimeError('Photo save failed')
                            self.last_saved=time.monotonic()
                            if self.count()>=self.target:self.collect=False
                    x1,y1,x2,y2=roi_bounds(frame.shape)
                    cv2.rectangle(preview,(x1,y1),(x2,y2),(0,210,40),2)
                    message=f'{self.cls}/{self.session}: {self.count()}/{self.target}' if self.mode=='collect' else ' | '.join(k+': '+v['top1'] for k,v in pred.items())
                    cv2.putText(preview,message,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,210,40),2)
                    if preview.shape[1]>640:
                        preview=cv2.resize(preview,(640,round(preview.shape[0]*640/preview.shape[1])))
                    ok,buf=cv2.imencode('.jpg',preview,[cv2.IMWRITE_JPEG_QUALITY,70])
                    if not ok:raise RuntimeError('Preview JPEG encode failed')
                    self.jpg=buf.tobytes()
                self.stop.wait(.01)
        except Exception as e:
            with self.lock:self.error=str(e);self.collect=False
            print('CAMERA ERROR:',e,flush=True)
    def image(self):
        with self.lock:
            if not self.jpg:raise RuntimeError(self.error or 'Waiting for first frame')
            return self.jpg
    def state(self):
        with self.lock:
            return {'mode':self.mode,'labels':self.classes,'sessions':self.sessions,'camera':self.cam.actual,'class':self.cls,'session':self.session,'saved':self.count() if self.mode=='collect' else None,
                    'target':self.target,'collecting':self.collect,'frame_sequence':self.sequence,'predictions':self.last,'error':self.error}
    def page(self,token):
        controls='<p>초록 ROI 안에 물체를 놓으세요. 원본 ROI만 저장하며 테두리·문자는 저장하지 않습니다.</p>'
        if self.mode=='collect':
            controls+='<select id="cls">'+''.join('<option>'+html.escape(c)+'</option>' for c in self.classes)+'</select>'
            controls+='<select id="session">'+''.join('<option>'+html.escape(s)+'</option>' for s in self.sessions)+'</select>'
            controls+="<button onclick=\"act('/select',{class:document.getElementById('cls').value,session:document.getElementById('session').value})\">선택 적용</button><button onclick=\"act('/start')\">자동 촬영 시작</button><button onclick=\"act('/stop')\">촬영 중지</button>"
        if self.mode=='imagenet':
            controls+='<p>현재 예측 표본 기록: 실제 정답과 조건을 고른 뒤 기록합니다. 학습 사진을 재사용하지 않습니다. ImageNet 분류명과 사용자 라벨은 서로 다르므로, 대응표를 명시한 클래스만 ImageNet 대응 점수를 계산합니다.</p><select id="truth">'+''.join('<option>'+html.escape(c)+'</option>' for c in self.truths)+'</select>'
            controls+='<input id="condition" value="condition_1" maxlength="80">'
            controls+="<button onclick=\"act('/record',{truth:document.getElementById('truth').value,condition:document.getElementById('condition').value})\">현재 프레임을 비교 표본으로 기록</button>"
        return page_html('Edge AI — '+self.mode,controls,token)
    def action(self,path,data):
        import cv2
        with self.lock:
            if self.error:raise RuntimeError(self.error)
            if path=='/select' and self.mode=='collect':
                c=data.get('class');s=data.get('session')
                if c not in self.classes or s not in self.sessions:raise ValueError('Unknown class/session')
                self.collect=False;self.cls=c;self.session=s
            elif path=='/start' and self.mode=='collect':self.collect=self.count()<self.target
            elif path=='/stop':self.collect=False
            elif path=='/record' and self.mode=='imagenet':
                truth=data.get('truth');condition=str(data.get('condition','')).strip()
                if truth not in self.truths or not condition or len(condition)>80:raise ValueError('Invalid truth/condition')
                if self.raw is None or not self.last:raise RuntimeError('No prediction yet')
                if self.sequence==self.last_recorded_sequence:raise ValueError('This frame was already recorded. Wait for a new frame.')
                folder=ROOT/'day2_extra/comparison';(folder/'images').mkdir(parents=True,exist_ok=True)
                name=uuid.uuid4().hex+'.jpg'
                if not cv2.imwrite(str(folder/'images'/name),self.raw):raise RuntimeError('Cannot record image')
                row={'image':name,'truth':truth,'condition':condition,'frame_sequence':self.sequence,
                     'hostname':metadata()['hostname'],'machine':metadata()['machine'],'threads':self.threads,
                     'timing_scope':'prepared_input_to_numpy_output; one live CPU prediction'}
                for key in ['custom','imagenet']:
                    value=self.last.get(key,{})
                    row[key+'_top1']=value.get('top1','');row[key+'_confidence']=value.get('confidence','')
                    row[key+'_top5']='|'.join(v['label'] for v in value.get('top5',[]))
                    row[key+'_inference_ms']=value.get('inference_ms','')
                    row[key+'_execution_path']=getattr(self.runners.get(key),'execution_path','')
                    row[key+'_model']=self.model_info.get(key,{}).get('model','')
                    row[key+'_model_sha256']=self.model_info.get(key,{}).get('model_sha256','')
                path=folder/'samples.csv';new=not path.exists()
                fieldnames=list(row)
                if not new:
                    with path.open(newline='',encoding='utf-8-sig') as f:
                        reader=csv.DictReader(f);old_fields=reader.fieldnames or []
                        if not set(fieldnames).issubset(old_fields):
                            previous=list(reader)
                        else:
                            previous=None;fieldnames=old_fields
                    if previous is not None:
                        # Upgrade old CSV columns without losing existing observations.
                        fieldnames=list(dict.fromkeys(old_fields+fieldnames))
                        import shutil
                        backup=path.with_name('samples_before_r2_'+uuid.uuid4().hex[:8]+'.csv')
                        shutil.copy2(path,backup)
                        tmp=path.with_suffix('.csv.tmp')
                        with tmp.open('w',newline='',encoding='utf-8') as f:
                            writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(previous)
                        tmp.replace(path)
                with path.open('a',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=fieldnames)
                    if new:w.writeheader()
                    w.writerow(row)
                self.last_recorded_sequence=self.sequence
            else:raise ValueError('Unsupported action')
        return {'ok':True}
    def close(self):
        self.stop.set();self.worker.join(timeout=3);self.cam.close()

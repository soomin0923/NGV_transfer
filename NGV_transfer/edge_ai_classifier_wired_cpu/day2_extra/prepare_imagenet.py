"""Prepare local pretrained ImageNet TFLite models on CPU; never train/download."""
from pathlib import Path
import sys, os, argparse, json, tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import numpy as np
from PIL import Image
from extra_common.tf_util import tf, convert_frozen, predict_array
from extra_common.runtime import LiteRunner
from extra_common.util import ROOT, write_json, finite, sha256
from extra_common.offline import require_asset
from extra_common.calibration import load_training_calibration


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--precision',choices=['fp32','dynamic','int8','all'],default='fp32')
    p.add_argument('--calibration-dir',default='dataset_split/train')
    p.add_argument('--calibration-images',type=int,default=128)
    p.add_argument('--sample-dir',default='dataset_split/val',help='Parity diagnostics only; never calibration')
    a=p.parse_args()
    modes=['fp32','dynamic','int8'] if a.precision=='all' else [a.precision]
    out=ROOT/'day2_extra/artifacts';out.mkdir(parents=True,exist_ok=True)
    m=tf.keras.applications.MobileNetV3Small(weights=None,include_top=True,include_preprocessing=True)
    weight_path=require_asset('weights/mobilenet_v3_small_top.h5');m.load_weights(str(weight_path))
    doc=json.loads(require_asset('imagenet_class_index.json').read_text())
    names=[doc[str(i)][1] for i in range(1000)]
    calibration=None;cal_doc=None
    if 'int8' in modes:
        calibration,cal_doc=load_training_calibration(a.calibration_dir,image_size=(224,224),max_images=a.calibration_images)
    paths=sorted(Path(a.sample_dir).rglob('*.jpg'))[:32]
    if paths:
        x=np.stack([np.asarray(Image.open(f).convert('RGB').resize((224,224),Image.Resampling.BILINEAR),np.float32) for f in paths])
        parity_source='local validation photos (no ImageNet ground-truth benchmark)'
    else:
        x=np.random.default_rng(42).uniform(0,255,(2,224,224,3)).astype(np.float32)
        parity_source='synthetic diagnostic inputs; NOT accuracy data'
    expected=finite(predict_array(m,x,8))
    marker=out/'imagenet_export_incomplete.json';write_json(marker,{'status':'incomplete','requested':modes})
    rows=[]
    with tempfile.TemporaryDirectory(prefix='.imagenet_stage_',dir=out) as temp:
        stage=Path(temp)
        for mode in modes:
            path=stage/f'imagenet_mobilenetv3small_{mode}.tflite'
            path.write_bytes(convert_frozen(m,[None,224,224,3],mode,calibration))
            runner=LiteRunner(path)
            actual=finite(np.concatenate([runner.predict(sample[None]) for sample in x]))
            if actual.shape!=expected.shape:raise ValueError('ImageNet output shape mismatch')
            is_int8=runner.input['dtype']==np.int8 and runner.output['dtype']==np.int8
            if mode=='int8' and not is_int8:raise ValueError('Requested full INT8 input/output was not produced')
            close=bool(np.allclose(actual,expected,atol=2e-4,rtol=2e-3))
            if mode=='fp32' and not close:raise ValueError('ImageNet FP32 parity failed')
            rows.append({'precision':mode,'model':path.name,'sha256':sha256(path),'bytes':path.stat().st_size,
                         'status':'PASS' if close else 'QUANTIZED_DIAGNOSTIC_DRIFT','finite':True,
                         'input_dtype':str(runner.input['dtype']),'output_dtype':str(runner.output['dtype']),
                         'execution_path':runner.execution_path,'delegate_fallback_reason':runner.delegate_fallback_reason,
                         'input_quantization':list(runner.input['quantization']),
                         'output_quantization':list(runner.output['quantization']),
                         'max_abs_error':float(np.max(np.abs(actual-expected))),
                         'top1_agreement_not_accuracy':float(np.mean(actual.argmax(1)==expected.argmax(1))),
                         'zero_output_rows':int(np.sum(actual.sum(1)==0))})
            del runner
        for row in rows:(stage/row['model']).replace(out/row['model'])
    (out/'labels.txt').write_text('\n'.join(names)+'\n',encoding='utf-8')
    if cal_doc is not None:write_json(out/'imagenet_calibration_manifest.json',cal_doc)
    # Keep independently prepared modes in the manifest only when bytes still match.
    previous=[]
    if (out/'metadata.json').exists():
        prior=json.loads((out/'metadata.json').read_text());previous=prior.get('models',[])
    retained=[r for r in previous if r.get('precision') not in modes and (out/r.get('model','')).is_file() and sha256(out/r['model'])==r.get('sha256')]
    metadata={'status':'validated','model':'MobileNetV3Small','classes':1000,'trained_in_this_lab':False,
              'preprocessing':'RGB 0..255; internal preprocessing','source':'Keras ImageNet pretrained weights',
              'pretrained_sha256':sha256(weight_path),'models':retained+rows,'parity_source':parity_source,
              'quantization_note':'INT8 ranges use local custom-camera TRAIN photos. This does not establish accuracy over all 1000 ImageNet classes. Quantized 1000-way softmax may round small probabilities to zero; low-confidence/tied outputs require caution.'}
    write_json(out/'metadata.json',metadata);marker.unlink()
    for row in rows:print(row)
    print('PASS: local pretrained models prepared without fit/fine-tuning. Quantized agreement is NOT labelled accuracy.')


if __name__=='__main__':main()

from pathlib import Path
import argparse, importlib, importlib.metadata as md, sys, platform, os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['CUDA_VISIBLE_DEVICES']='-1'
p=argparse.ArgumentParser();p.add_argument('--role',choices=['train','tools','pi'],required=True);a=p.parse_args()
print('ROLE:',a.role,'Python:',sys.executable,'CPU:',platform.machine())
assert sys.prefix!=sys.base_prefix,'Activate the role-specific venv first'
mods={'train':['tensorflow','ai_edge_litert.interpreter','numpy','PIL','sklearn','matplotlib'],
      'tools':['onnx','onnxruntime','numpy'],'pi':['ai_edge_litert.interpreter','cv2','numpy','PIL']}[a.role]
for m in mods:
    module=importlib.import_module(m);print('IMPORT PASS:',m,getattr(module,'__version__',''))
pins={'train':{'tensorflow-cpu':'2.21.0','keras':'3.15.1','ai-edge-litert':'2.2.0','numpy':'2.2.6'},
      'tools':{'onnx':'1.18.0','onnxruntime':'1.22.1','protobuf':'5.29.5','numpy':'2.2.6'},
      'pi':{'ai-edge-litert':'2.2.0'}}[a.role]
for package,version in pins.items():
    actual=md.version(package)
    if actual!=version:raise SystemExit(f'VERSION FAIL: {package} {actual} != {version}')
if a.role=='train':
    import tensorflow as tf
    assert not tf.config.get_visible_devices('GPU')
    print('CPU only:',tf.config.get_visible_devices('CPU'))
if a.role=='tools':
    import onnxruntime as ort
    assert 'CPUExecutionProvider' in ort.get_available_providers()
    print('Course sessions explicitly use CPUExecutionProvider.')
if a.role=='pi':assert platform.machine()=='aarch64','Pi OS must be 64-bit aarch64'
print('PASS: imports and pinned core versions. Camera/SSH require the actual classroom hardware.')

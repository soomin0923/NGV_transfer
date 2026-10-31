"""CPU MobileNetV3 transfer learning using local, instructor-prepared weights."""
from pathlib import Path
import sys, argparse, shutil, time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',default='dataset_split');p.add_argument('--output',default='artifacts')
p.add_argument('--architecture',choices=['mobilenetv3'],default='mobilenetv3')
p.add_argument('--epochs',type=int,default=5);p.add_argument('--fine-tune-epochs',type=int,default=1)
p.add_argument('--batch-size',type=int,default=16);p.add_argument('--image-size',type=int,default=160)
p.add_argument('--weights',default=None,help='Optional local pretrained no_top HDF5; never downloads')
p.add_argument('--skip-export',action='store_true');p.add_argument('--allow-cpu',action='store_true',help='Compatibility flag; always CPU')
a=p.parse_args()
if a.epochs<1 or a.fine_tune_epochs<0 or a.batch_size<1 or a.image_size<32: p.error('Invalid epochs/batch/image size')
from extra_common.tf_util import tf,setup_training,compile_model
from extra_common.util import labels,write_json,metadata,sha256
from extra_common.offline import require_asset
from extra_common.course import load_config
setup_training(); started=time.perf_counter()
weights=Path(a.weights) if a.weights else require_asset('weights/mobilenet_v3_small_notop.h5')
if not weights.is_file(): raise SystemExit('Missing local weights. Run tools/prepare_offline.py in advance.')
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);names=labels(Path(a.data)/'class_names.txt')
if names!=load_config()['labels']:raise SystemExit('Split label order differs from course_config.json. Use the same config as capture/split.')
if any((out/name).exists() for name in ['best_model.keras','best_head.keras','best_fine_tuned.keras']):
    raise SystemExit('STOP: model exists. Use a fresh --output directory for a new run; do not mix results.')
datasets={}
for split in ['train','val','test']:
    ds=tf.keras.utils.image_dataset_from_directory(Path(a.data)/split,class_names=names,label_mode='int',image_size=(a.image_size,a.image_size),batch_size=a.batch_size,shuffle=(split=='train'),seed=42)
    opt=tf.data.Options();opt.threading.private_threadpool_size=2;datasets[split]=ds.with_options(opt).prefetch(1)
base=tf.keras.applications.MobileNetV3Small(input_shape=(a.image_size,a.image_size,3),include_top=False,weights=None,include_preprocessing=True)
base.load_weights(str(weights));base.trainable=False
inp=tf.keras.Input((a.image_size,a.image_size,3),name='rgb_0_255')
x=tf.keras.layers.RandomFlip('horizontal')(inp);x=tf.keras.layers.RandomRotation(.05)(x)
x=base(x,training=False);x=tf.keras.layers.GlobalAveragePooling2D()(x);x=tf.keras.layers.Dropout(.2)(x)
model=tf.keras.Model(inp,tf.keras.layers.Dense(len(names),activation='softmax')(x));compile_model(model)
def fit_stage(path,epochs):
    return model.fit(datasets['train'],validation_data=datasets['val'],epochs=epochs,callbacks=[tf.keras.callbacks.ModelCheckpoint(path,monitor='val_accuracy',mode='max',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',patience=3,restore_best_weights=True)]).history
history={'head':fit_stage(out/'best_head.keras',a.epochs)};candidates=['best_head']
if a.fine_tune_epochs:
    model=tf.keras.models.load_model(out/'best_head.keras')
    base=next(layer for layer in model.layers if isinstance(layer,tf.keras.Model));base.trainable=True
    for layer in base.layers[:-30]:layer.trainable=False
    for layer in base.layers:
        if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
    compile_model(model,1e-5)
    history['fine_tuned']=fit_stage(out/'best_fine_tuned.keras',a.fine_tune_epochs)
    candidates.append('best_fine_tuned')
scores={}
for name in candidates:
    m=tf.keras.models.load_model(out/(name+'.keras'));scores[name]=float(m.evaluate(datasets['val'],verbose=0)[1])
best=max(scores,key=scores.get);shutil.copy2(out/(best+'.keras'),out/'best_model.keras')
(out/'labels.txt').write_text('\n'.join(names)+'\n',encoding='utf-8')
write_json(out/'training_report.json',{'environment':metadata(),'device':'CPU','seconds':time.perf_counter()-started,'validation_accuracy':scores,'selected':best,'history':history,'labels':names,'image_size':a.image_size,'batch_size':a.batch_size,'pretrained_sha256':sha256(weights),'preprocessing':'RGB float32 0..255; MobileNetV3 internal preprocessing','architecture':'MobileNetV3Small'})
print('PASS: CPU best_model.keras and labels.txt. Selected:',best)
if not a.skip_export:print('Next separate CPU process: python wsl/export_frozen.py --float32-only --sample-dir dataset_split/val')

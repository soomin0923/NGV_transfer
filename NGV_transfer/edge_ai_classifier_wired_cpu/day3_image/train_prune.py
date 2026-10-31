from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=5);p.add_argument('--finetune-epochs',type=int,default=2)
p.add_argument('--keep-ratio',type=float,default=.5);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--eval-batch-size',type=int,default=128);p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'offline/data/fashion-mnist.npz');p.add_argument('--train-samples',type=int,default=12000);p.add_argument('--val-samples',type=int,default=3000);p.add_argument('--test-samples',type=int,default=3000);p.add_argument('--full-data',action='store_true');p.add_argument('--seed',type=int,default=42);a=p.parse_args()
import numpy as np
from extra_common.tf_util import tf,setup_training,compile_model,dataset,predict_array
from extra_common.util import ROOT,write_json,metadata,finite,sha256
from day3.numerics import save_weights
from day3_image.numerics import prune_cnn,LABELS
from day3_image.data import load_local,split_indices
if a.epochs<1 or a.finetune_epochs<1 or a.batch_size<1 or a.eval_batch_size<1:raise SystemExit('Epoch and batch sizes must be positive')
if not 0<a.keep_ratio<=1:raise SystemExit('--keep-ratio must be in (0,1]')
setup_training(seed=a.seed);out=ROOT/'day3_image/artifacts';out.mkdir(parents=True,exist_ok=True)
if (out/'training_report.json').exists():raise SystemExit('STOP: completed Fashion-MNIST training exists. Do not overwrite a finished comparison.')
data=load_local(a.data)
x,y,xt,yt=(data[k] for k in ['x_train','y_train','x_test','y_test'])
indices=split_indices(y,yt,a.train_samples,a.val_samples,a.test_samples,a.seed,a.full_data)
ti,vi,eval_indices=indices['train'],indices['val'],indices['test']
trainx=(x[ti,...,None]/255.).astype(np.float32);valx=(x[vi,...,None]/255.).astype(np.float32)
evalx=(xt[eval_indices,...,None]/255.).astype(np.float32);evaly=yt[eval_indices].astype(np.int64)
rng=np.random.default_rng(a.seed);cal_indices=rng.choice(len(trainx),min(256,len(trainx)),replace=False)
np.savez_compressed(out/'evaluation.npz',x=evalx,y=evaly,test_indices=eval_indices)
np.savez_compressed(out/'calibration.npz',x=trainx[cal_indices],train_source_indices=ti[cal_indices])
np.savez_compressed(out/'split_indices.npz',train_source_indices=ti,val_source_indices=vi,test_source_indices=eval_indices)
print(f'CPU local data: train={len(ti)}, val={len(vi)}, official-test={len(eval_indices)}',flush=True)
(out/'labels.txt').write_text('\n'.join(LABELS)+'\n',encoding='utf-8')
train=dataset(trainx,y[ti],a.batch_size,True);val=dataset(valx,y[vi],a.eval_batch_size)
def make(c1=32,c2=64,hidden=128):
    return tf.keras.Sequential([tf.keras.Input((28,28,1),name='input'),tf.keras.layers.Conv2D(c1,3,padding='same',activation='relu',name='conv1'),tf.keras.layers.MaxPooling2D(2),tf.keras.layers.Conv2D(c2,3,padding='same',activation='relu',name='conv2'),tf.keras.layers.MaxPooling2D(2),tf.keras.layers.Flatten(),tf.keras.layers.Dense(hidden,activation='relu',name='dense1'),tf.keras.layers.Dense(10,activation='softmax',name='probabilities')])
def accuracy(m):return float(np.mean(predict_array(m,evalx,a.eval_batch_size).argmax(1)==evaly))
def save(m,name):
    for w in m.get_weights():finite(w)
    m.save(out/(name+'.keras'));save_weights(out/(name+'.weights.npz'),m.get_weights())
    # Validation samples, not calibration/test, establish export parity.
    np.savez_compressed(out/(name+'.reference.npz'),x=valx[:64],y=predict_array(m,valx[:64],a.eval_batch_size))
base=compile_model(make());hist=base.fit(train,validation_data=val,epochs=a.epochs).history;save(base,'baseline');base_acc=accuracy(base)
pw,selected=prune_cnn(base.get_weights(),a.keep_ratio);pruned=make(len(pw[1]),len(pw[3]),len(pw[5]));pruned.set_weights(pw)
before=accuracy(pruned);compile_model(pruned,1e-4);phist=pruned.fit(train,validation_data=val,epochs=a.finetune_epochs).history;save(pruned,'pruned');after=accuracy(pruned)
control=make();control.set_weights(base.get_weights());compile_model(control,1e-4);chist=control.fit(train,validation_data=val,epochs=a.finetune_epochs).history;save(control,'dense_control')
report={'environment':metadata(),'device':'CPU','seed':a.seed,'source_data':str(a.data.resolve()),'source_sha256':sha256(a.data),'full_data':a.full_data,'epochs':a.epochs,'finetune_epochs':a.finetune_epochs,'input':'28x28x1 float32 0..1','train_samples':len(ti),'val_samples':len(vi),'evaluation_samples':len(evalx),'full_test_samples':len(yt),'calibration_split':'train','baseline_accuracy':base_acc,'pruned_before_finetune_accuracy':before,'pruned_after_finetune_accuracy':after,'dense_control_extra_epochs_accuracy':accuracy(control),'baseline_parameters':int(base.count_params()),'pruned_parameters':int(pruned.count_params()),'selected_channels':selected,'history':{'baseline':hist,'pruned':phist,'dense_control':chist}}
write_json(out/'training_report.json',report);print('PASS: baseline, pruned and equal-extra-epoch dense control saved');print({k:v for k,v in report.items() if 'accuracy' in k})

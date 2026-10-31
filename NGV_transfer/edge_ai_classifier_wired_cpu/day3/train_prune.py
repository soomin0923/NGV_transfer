from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
t=sub.add_parser('train');t.add_argument('--epochs',type=int,default=10);t.add_argument('--batch-size',type=int,default=64)
t=sub.add_parser('prune');t.add_argument('--sparsity',type=float,default=.5);t.add_argument('--keep',type=float,default=.5);t.add_argument('--finetune-epochs',type=int,default=3);t.add_argument('--batch-size',type=int,default=64);a=p.parse_args()
if a.batch_size<1 or (a.action=='train' and a.epochs<1) or (a.action=='prune' and a.finetune_epochs<1):raise SystemExit('Epoch and batch sizes must be positive')
import numpy as np
from extra_common.tf_util import tf,setup_training,compile_model,dataset,predict_array
from extra_common.util import ROOT,write_json,finite,metadata
from day3.numerics import magnitude_mask,prune_mlp,save_weights,parameters
setup_training()
z=np.load(ROOT/'day3/out/data/dataset.npz',allow_pickle=False);out=ROOT/'day3/out/models';out.mkdir(parents=True,exist_ok=True)
train=dataset(z['x_train'],z['y_train'],a.batch_size,True);val=dataset(z['x_val'],z['y_val'],a.batch_size)
def make(h1=512,h2=256):
    return tf.keras.Sequential([tf.keras.Input((256,),name='features'),tf.keras.layers.Dense(h1,activation='relu',name='hidden1'),tf.keras.layers.Dense(h2,activation='relu',name='hidden2'),tf.keras.layers.Dense(4,activation='softmax',name='probabilities')])
def save(m,name,history,before=None):
    weights=m.get_weights()
    for w in weights:finite(w)
    m.save(out/(name+'.keras'));save_weights(out/(name+'.weights.npz'),weights)
    reference_x=z['x_val'][:128];np.savez_compressed(out/(name+'.reference.npz'),x=reference_x,y=predict_array(m,reference_x))
    pred=predict_array(m,z['x_test']);accuracy=float(np.mean(pred.argmax(1)==z['y_test']))
    write_json(out/(name+'.json'),{'variant':name,'accuracy':accuracy,'samples':len(pred),'parameters':parameters(weights),'dense_macs':sum(w.size for w in weights[::2]),'zero_fraction_kernels':float(sum(np.count_nonzero(w==0) for w in weights[::2])/sum(w.size for w in weights[::2])),'history':history,'pre_finetune_accuracy':before,'environment':metadata()});print(name,accuracy)
class MaskCallback(tf.keras.callbacks.Callback):
    def __init__(self,masks):super().__init__();self.masks=masks
    def apply(self):
        for layer,mask in zip([l for l in self.model.layers if isinstance(l,tf.keras.layers.Dense)],self.masks):layer.kernel.assign(layer.kernel*mask)
    def on_train_begin(self,logs=None):self.apply()
    def on_train_batch_end(self,batch,logs=None):self.apply()
    def on_train_end(self,logs=None):self.apply()
if a.action=='train':
    if (out/'baseline.keras').exists():raise SystemExit('STOP: baseline already exists')
    m=compile_model(make());hist=m.fit(train,validation_data=val,epochs=a.epochs,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True)]).history;save(m,'baseline',hist)
else:
    if any((out/(name+'.keras')).exists() for name in ['unstructured','structured']):raise SystemExit('STOP: pruning results already exist; archive day3/out before repeating a complete comparison')
    base=tf.keras.models.load_model(out/'baseline.keras');w=base.get_weights()
    m=make();m.set_weights(w);masks=[magnitude_mask(v,a.sparsity) for v in w[::2]]
    masked=[v*mask for v,mask in zip(w[::2],masks)];mw=[]
    for v,b in zip(masked,w[1::2]):mw.extend([v,b])
    m.set_weights(mw);before=float(np.mean(predict_array(m,z['x_test']).argmax(1)==z['y_test']));compile_model(m,1e-4)
    hist=m.fit(train,validation_data=val,epochs=a.finetune_epochs,callbacks=[MaskCallback(masks)]).history;save(m,'unstructured',hist,before)
    pw,_=prune_mlp(w,a.keep);m=make(len(pw[1]),len(pw[3]));m.set_weights(pw);before=float(np.mean(predict_array(m,z['x_test']).argmax(1)==z['y_test']));compile_model(m,1e-4)
    hist=m.fit(train,validation_data=val,epochs=a.finetune_epochs).history;save(m,'structured',hist,before)

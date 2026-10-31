"""Pure NumPy educational signal and pruning operations; no TensorFlow import."""
import numpy as np
LABELS=['normal','unbalance_like','misalignment_like','impact_like']
FS=2048;N=1024

def signal(kind,rng,frequency=40.,gain=1.,noise=.08):
    t=np.arange(N)/FS;phase=rng.uniform(0,2*np.pi);f=frequency*rng.uniform(.985,1.015)
    x=.8*np.sin(2*np.pi*f*t+phase)+.12*np.sin(4*np.pi*f*t)
    if kind==1:x+=1.8*np.sin(2*np.pi*f*t+phase)
    elif kind==2:x+=1.1*np.sin(4*np.pi*f*t+phase)+.65*np.sin(6*np.pi*f*t)
    elif kind==3:
        for start in rng.integers(0,N,size=8):
            dt=(np.arange(N-start))/FS
            x[start:]+=rng.uniform(1.5,2.5)*np.exp(-dt*110)*np.sin(2*np.pi*340*dt)
    elif kind!=0:raise ValueError('kind must be 0..3')
    return (gain*x+rng.normal(0,noise,N)).astype(np.float32)

def features(raw):
    raw=np.asarray(raw,np.float64);x=raw-raw.mean(axis=-1,keepdims=True)
    spectrum=np.abs(np.fft.rfft(x*np.hanning(N),axis=-1))[...,1:257]/(N/2)
    return np.log1p(spectrum).astype(np.float32)

def magnitude_mask(weight,sparsity):
    if not 0<=sparsity<1:raise ValueError('sparsity must be in [0,1)')
    mask=np.ones(weight.size,np.float32);idx=np.argsort(np.abs(weight).ravel(),kind='stable')
    mask[idx[:int(weight.size*sparsity)]]=0
    return mask.reshape(weight.shape)

def keep_indices(scores,ratio):
    if not 0<ratio<=1:raise ValueError('keep ratio must be in (0,1]')
    count=max(1,int(len(scores)*ratio))
    return np.sort(np.argsort(-np.asarray(scores),kind='stable')[:count])

def prune_mlp(weights,ratio):
    w1,b1,w2,b2,w3,b3=weights
    i=keep_indices(np.abs(w1).sum(axis=0),ratio)
    j=keep_indices(np.abs(w2[i]).sum(axis=0),ratio)
    return [w1[:,i],b1[i],w2[np.ix_(i,j)],b2[j],w3[j,:],b3.copy()],(i,j)

def mlp_numpy(x,weights):
    for i in range(0,len(weights),2):
        x=np.asarray(x)@weights[i]+weights[i+1]
        if i<len(weights)-2:x=np.maximum(x,0)
    x=x-x.max(axis=-1,keepdims=True);p=np.exp(x);return p/p.sum(axis=-1,keepdims=True)

def parameters(weights):return sum(w.size for w in weights)

def save_weights(path,weights):np.savez_compressed(path,**{f'w{i}':w for i,w in enumerate(weights)})
def load_weights(path):
    with np.load(path,allow_pickle=False) as z:return [z[f'w{i}'] for i in range(len(z.files))]

"""Structured channel pruning with correct NHWC Flatten index mapping."""
import numpy as np
from day3.numerics import keep_indices
LABELS=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
def flatten_channel_indices(height,width,channels,keep):
    return (np.arange(height*width)[:,None]*channels+np.asarray(keep)[None,:]).ravel()
def prune_cnn(weights,ratio):
    w1,b1,w2,b2,w3,b3,w4,b4=weights
    c1=keep_indices(np.abs(w1).sum(axis=(0,1,2)),ratio)
    c2=keep_indices(np.abs(w2[:,:,c1,:]).sum(axis=(0,1,2)),ratio)
    rows=flatten_channel_indices(7,7,len(b2),c2)
    hidden=keep_indices(np.abs(w3[rows]).sum(axis=0),ratio)
    return [w1[:,:,:,c1],b1[c1],w2[:,:,c1,:][:,:,:,c2],b2[c2],w3[np.ix_(rows,hidden)],b3[hidden],w4[hidden],b4.copy()],{'conv1':c1.tolist(),'conv2':c2.tolist(),'dense':hidden.tolist()}

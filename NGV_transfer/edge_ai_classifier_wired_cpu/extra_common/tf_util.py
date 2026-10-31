"""TensorFlow CPU-only training and conversion helpers."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
from extra_common.course import load_config
_CPU_THREADS=int(os.environ.get('EDGE_CPU_THREADS',load_config()['threads']))
os.environ['TF_NUM_INTRAOP_THREADS']=str(_CPU_THREADS)
os.environ['TF_NUM_INTEROP_THREADS']='1'
import numpy as np
import tensorflow as tf

def setup_training(seed=42,allow_cpu=True):
    # CPU-only even when a GPU is present; set before importing TensorFlow above.
    tf.config.set_visible_devices([], 'GPU')
    tf.keras.utils.set_random_seed(seed)
    tf.keras.mixed_precision.set_global_policy('float32')
    if tf.config.get_visible_devices('GPU'):
        raise RuntimeError('CPU-only device validation failed')
    print('Training: CPU only; threads =', _CPU_THREADS)
    return []

def freeze(model,shape):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    @tf.function(input_signature=[tf.TensorSpec(shape,tf.float32,name='input')],autograph=False)
    def serve(x):return model(x,training=False)
    return convert_variables_to_constants_v2(serve.get_concrete_function())

def convert_frozen(model,shape,mode='fp32',calibration=None):
    frozen=freeze(model,shape)
    converter=tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    if mode=='dynamic': converter.optimizations=[tf.lite.Optimize.DEFAULT]
    elif mode=='int8':
        if calibration is None or len(calibration)==0: raise ValueError('INT8 needs training-only representative data')
        converter.optimizations=[tf.lite.Optimize.DEFAULT]
        converter.representative_dataset=lambda: ([np.asarray(x[None],np.float32)] for x in calibration)
        converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type=tf.int8; converter.inference_output_type=tf.int8
    elif mode!='fp32':raise ValueError(mode)
    return converter.convert()

def predict_array(model,x,batch=64):
    return np.concatenate([np.asarray(model(x[i:i+batch],training=False)) for i in range(0,len(x),batch)])

def dataset(x,y,batch=64,shuffle=False):
    ds=tf.data.Dataset.from_tensor_slices((x,y))
    if shuffle:ds=ds.shuffle(len(x),seed=42,reshuffle_each_iteration=True)
    opt=tf.data.Options(); opt.threading.private_threadpool_size=4
    return ds.batch(batch).with_options(opt).prefetch(1)

def compile_model(model,lr=1e-3):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'],jit_compile=False)
    return model

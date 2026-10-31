"""Structure-specific Keras-weight -> ONNX exporters. They are not generic converters."""
from pathlib import Path
import numpy as np

def save_graph(nodes,initializers,shape,path):
    import onnx
    from onnx import helper,TensorProto
    classes=int(initializers[-1].dims[0]) if False else None
    graph=helper.make_graph(nodes,'edge_course',[helper.make_tensor_value_info('input',TensorProto.FLOAT,shape)],
        [helper.make_tensor_value_info('output',TensorProto.FLOAT,['batch',4 if len(shape)==2 else 10])],initializers)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],producer_name='edge_direct_course')
    model.ir_version=10;onnx.checker.check_model(model);Path(path).parent.mkdir(parents=True,exist_ok=True)
    onnx.save(onnx.shape_inference.infer_shapes(model),str(path))

def mlp_onnx(weights,path):
    from onnx import helper,numpy_helper
    nodes=[];initializers=[];current='input'
    for i in range(3):
        w,b=weights[2*i:2*i+2];initializers.extend([numpy_helper.from_array(w.astype(np.float32),f'w{i}'),numpy_helper.from_array(b.astype(np.float32),f'b{i}')])
        nodes.append(helper.make_node('MatMul',[current,f'w{i}'],[f'mm{i}'],name=f'MatMul_{i}'))
        nodes.append(helper.make_node('Add',[f'mm{i}',f'b{i}'],[f'add{i}'],name=f'Bias_{i}'))
        current=f'act{i}' if i<2 else 'output';nodes.append(helper.make_node('Relu' if i<2 else 'Softmax',[f'add{i}'],[current],name=f'Activation_{i}',**({'axis':1} if i==2 else {})))
    save_graph(nodes,initializers,['batch',256],path)

def cnn_onnx(weights,path):
    from onnx import helper,numpy_helper
    w1,b1,w2,b2,w3,b3,w4,b4=weights;init=[];nodes=[helper.make_node('Transpose',['input'],['nchw'],perm=[0,3,1,2],name='Input_NHWC_to_NCHW')]
    for name,value in [('c1w',w1.transpose(3,2,0,1)),('c1b',b1),('c2w',w2.transpose(3,2,0,1)),('c2b',b2),('d1w',w3),('d1b',b3),('d2w',w4),('d2b',b4)]:init.append(numpy_helper.from_array(value.astype(np.float32),name))
    current='nchw'
    for i in [1,2]:
        nodes.extend([helper.make_node('Conv',[current,f'c{i}w',f'c{i}b'],[f'conv{i}'],kernel_shape=[3,3],pads=[1,1,1,1],name=f'Conv_{i}'),helper.make_node('Relu',[f'conv{i}'],[f'relu{i}'],name=f'Relu_{i}'),helper.make_node('MaxPool',[f'relu{i}'],[f'pool{i}'],kernel_shape=[2,2],strides=[2,2],name=f'Pool_{i}')]);current=f'pool{i}'
    # Keras Flatten is NHWC: transpose back BEFORE flatten, never blindly flatten NCHW.
    nodes.extend([helper.make_node('Transpose',[current],['nhwc'],perm=[0,2,3,1],name='Restore_NHWC'),helper.make_node('Flatten',['nhwc'],['flat'],axis=1,name='Flatten')])
    nodes.extend([helper.make_node('MatMul',['flat','d1w'],['mm1'],name='Dense_1'),helper.make_node('Add',['mm1','d1b'],['add1'],name='DenseBias_1'),helper.make_node('Relu',['add1'],['dense_relu'],name='DenseRelu'),helper.make_node('MatMul',['dense_relu','d2w'],['mm2'],name='Dense_2'),helper.make_node('Add',['mm2','d2b'],['logits'],name='DenseBias_2'),helper.make_node('Softmax',['logits'],['output'],axis=1,name='Softmax')])
    save_graph(nodes,init,['batch',28,28,1],path)

def qdq_quantize(source,destination,calibration,ops=None):
    from onnxruntime.quantization import quantize_static,CalibrationDataReader,QuantFormat,QuantType,CalibrationMethod
    class Reader(CalibrationDataReader):
        def __init__(self):self.rewind()
        def get_next(self):return next(self.items,None)
        def rewind(self):self.items=iter({'input':np.asarray(x[None],np.float32)} for x in calibration)
    quantize_static(str(source),str(destination),Reader(),quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,op_types_to_quantize=ops or ['Conv','MatMul'],
        extra_options={'ActivationSymmetric':True,'WeightSymmetric':True,'QuantizeBias':False,'QDQOpTypePerChannelSupportToAxis':{'MatMul':1}})
    # Retain FP32 bias while quantizing supported weights/activations for CPU QDQ.
    import onnx
    model=onnx.load(str(destination));initial={x.name:x for x in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type=='DequantizeLinear' and node.input[0] in initial:
            if initial[node.input[0]].data_type==onnx.TensorProto.INT32:raise ValueError('Unexpected INT32 DequantizeLinear; QuantizeBias=False was not honored')
    onnx.checker.check_model(model)

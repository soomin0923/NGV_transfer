"""LiteRT and ONNX CPU adapters. Quantization uses scale AND zero-point."""
from __future__ import annotations
from pathlib import Path
import threading, time
import numpy as np

def quantize_input(x, detail):
    dtype=np.dtype(detail['dtype']); x=np.asarray(x,dtype=np.float32)
    if not np.isfinite(x).all(): raise ValueError('Input has NaN/Inf')
    if np.issubdtype(dtype,np.integer):
        scale,zp=detail['quantization']
        if scale<=0: raise ValueError('Invalid input quantization scale')
        lim=np.iinfo(dtype)
        return np.ascontiguousarray(np.clip(np.rint(x/scale+zp),lim.min,lim.max).astype(dtype))
    return np.ascontiguousarray(x.astype(dtype))

def dequantize_output(x, detail):
    if np.issubdtype(np.dtype(detail['dtype']),np.integer):
        scale,zp=detail['quantization']
        if scale<=0: raise ValueError('Invalid output quantization scale')
        x=(x.astype(np.float32)-zp)*scale
    x=np.asarray(x,dtype=np.float32)
    if not np.isfinite(x).all(): raise ValueError('Model output contains NaN/Inf. Do not deploy this model.')
    return x

class LiteRunner:
    def __init__(self, model, threads=4):
        if threads < 1: raise ValueError('threads must be >=1')
        if not Path(model).is_file(): raise FileNotFoundError(f'Missing model: {model}. Transfer the exported model first.')
        from ai_edge_litert.interpreter import Interpreter
        self.model = str(model)
        self.threads = threads
        self.lock = threading.Lock()
        self.last_invoke_ms = 0.
        self.execution_path = 'xnnpack_or_default_cpu'
        self.delegate_fallback_reason = None
        self.interpreter = Interpreter(model_path=self.model, num_threads=threads)
        try:
            self.interpreter.allocate_tensors()
        except RuntimeError as exc:
            if not self._can_use_integer_builtin(exc): raise
            self._activate_integer_builtin(None, exc)
        self._refresh()

    def _can_use_integer_builtin(self, exc):
        # A delegate runtime failure is not a model-format or data-shape failure.
        # Preserve the exact INT8 model; never substitute a float model.
        details = self.interpreter.get_input_details()
        message = str(exc).lower()
        return (self.execution_path != 'builtin_cpu_no_default_delegates'
                and len(details) == 1 and details[0]['dtype'] == np.int8
                and ('xnnpack' in message or 'delegate' in message))

    def _activate_integer_builtin(self, shape, exc):
        import warnings
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        replacement = Interpreter(model_path=self.model, num_threads=self.threads,
                                  experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        if shape is not None:
            inp = replacement.get_input_details()[0]
            replacement.resize_tensor_input(inp['index'], list(shape), strict=True)
        replacement.allocate_tensors()
        self.interpreter = replacement
        self.execution_path = 'builtin_cpu_no_default_delegates'
        self.delegate_fallback_reason = str(exc)
        warnings.warn('INT8 CPU delegate unavailable; using builtin CPU integer kernels with the SAME INT8 model. '
                      'No floating-point model fallback. Reason: ' + str(exc), RuntimeWarning, stacklevel=2)

    def _refresh(self):
        ins = self.interpreter.get_input_details(); outs = self.interpreter.get_output_details()
        if len(ins) != 1 or len(outs) != 1: raise ValueError('This course expects one input and one output tensor')
        self.input = ins[0]; self.output = outs[0]

    def predict(self, x):
        with self.lock:
            if tuple(x.shape) != tuple(self.input['shape']):
                self.interpreter.resize_tensor_input(self.input['index'], list(x.shape), strict=True)
                try:
                    self.interpreter.allocate_tensors()
                except RuntimeError as exc:
                    if not self._can_use_integer_builtin(exc): raise
                    self._activate_integer_builtin(x.shape, exc)
                self._refresh()
            q = quantize_input(x, self.input)
            self.interpreter.set_tensor(self.input['index'], q)
            t = time.perf_counter_ns()
            try:
                self.interpreter.invoke()
            except RuntimeError as exc:
                if not self._can_use_integer_builtin(exc): raise
                self._activate_integer_builtin(x.shape, exc)
                self._refresh()
                self.interpreter.set_tensor(self.input['index'], quantize_input(x, self.input))
                t = time.perf_counter_ns()
                self.interpreter.invoke()
            self.last_invoke_ms = (time.perf_counter_ns()-t)/1e6
            return dequantize_output(self.interpreter.get_tensor(self.output['index']), self.output)

class OnnxRunner:
    def __init__(self,model,threads=4):
        import onnxruntime as ort
        opts=ort.SessionOptions(); opts.intra_op_num_threads=threads; opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
        self.name=self.session.get_inputs()[0].name
    def predict(self,x):
        y=self.session.run(None,{self.name:np.ascontiguousarray(x,dtype=np.float32)})[0]
        if not np.isfinite(y).all(): raise ValueError('ONNX output has NaN/Inf')
        return y

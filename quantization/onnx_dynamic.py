import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'fp32_kpd.onnx'
model_quant = 'int8_kpd.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)


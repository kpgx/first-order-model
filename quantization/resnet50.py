import torchvision.models as models
import torch
import torch.onnx

resnet50 = models.resnet50(pretrained=True, progress=False).eval()

BATCH_SIZE = 32

dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)

torch.onnx.export(resnet50, dummy_input, "resnet50_pytorch.onnx", verbose=False)

from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import numpy as np

url='https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg'
img = resize(io.imread(url), (224, 224))
img = np.expand_dims(np.array(img, dtype=np.float32), axis=0) # Expand image to have a batch dimension
input_batch = np.array(np.repeat(img, BATCH_SIZE, axis=0), dtype=np.float32) # Repeat across the batch dimension

resnet50_gpu = models.resnet50(pretrained=True, progress=False).to("cuda").eval()

input_batch_chw = torch.from_numpy(input_batch).transpose(1,3).transpose(2,3)
input_batch_gpu = input_batch_chw.to("cuda")

with torch.no_grad():
    predictions = np.array(resnet50_gpu(input_batch_gpu).cpu())

resnet50_gpu_half = resnet50_gpu.half()
input_half = input_batch_gpu.half()

with torch.no_grad():
    preds = np.array(resnet50_gpu_half(input_half).cpu()) # Warm Up

with torch.no_grad():
    preds = np.array(resnet50_gpu_half(input_half).cpu())



indices = (-predictions[0]).argsort()[:5]
print("Class | Likelihood")
print(list(zip(indices, predictions[0][indices])))



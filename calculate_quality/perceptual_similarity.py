import torch

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
img1 = torch.rand(10, 3, 100, 100)
img2 = torch.rand(10, 3, 100, 100)
print(lpips(img1, img2))

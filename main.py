import os
import cv2
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import colour
import numpy as np
import colour_demosaicing
from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from mosaicking import mosaicking
from downSampling import downSamplingRGB, downSamplingW
from demosaick_rgb import demosaick_rgb
import setNoisy
import upSampling
# 设置字体的属性
# plt.rcParams["font.sans-serif"] = "Arial Unicode MS"
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# 读入 rgbw 四通道图像
inputImage = scipy.io.loadmat('balloons_ms.mat')
inputImage = inputImage['rgbw'] / 255

# 提取 rgb 通道和 w 通道，并进行归一化，否则 bug
rgb = inputImage[:, :, :3]
W = inputImage[:, :, 3]

# 马赛克化
raw = mosaicking(rgb, 'Hamilton')

# raw = setNoisy(raw)

# 下采样
quadBayer = downSamplingRGB(raw, 'Mean')
W2 = downSamplingW(W, 'Mean')

# 引导滤波去马赛克
outputImage = demosaick_rgb(quadBayer, W2)
# outputImage = outputImage[0:inputImage.shape[0], 0:inputImage.shape[1]] # 裁剪输出图像，使其与输入图像尺寸相同，以便计算 PSNR

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes[0, 0].imshow(rgb)
axes[0, 0].set_title('输入图像的RGB通道')
axes[0, 0].axis('off')
axes[0, 1].imshow(outputImage)
axes[0, 1].set_title('输出图像（半成品）')
axes[0, 1].axis('off')
plt.tight_layout()
plt.show()

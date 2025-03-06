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
# 设置字体的属性
# plt.rcParams["font.sans-serif"] = "Arial Unicode MS"
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# 马赛克化
def mosaicking(image):
    pcfa = np.zeros((8, 8, 3)) # 设置单个彩色滤波阵列
    pcfa[:, :, 0] = [[1, 1, 1, 1, 0, 0, 0, 0], # R通道
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0], ]
    pcfa[:, :, 1] = [[0, 0, 0, 0, 1, 1, 1, 1], # G通道
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0], ]
    pcfa[:, :, 2] = [[0, 0, 0, 0, 0, 0, 0, 0], # B通道
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1, 1], ]

    r = pcfa.shape[0] # 单个 CFA 尺寸
    c = pcfa.shape[1]
    rr = image.shape[0] # 图片尺寸
    cc = image.shape[1]
    r_ceil = int(np.ceil(rr / r)) # CFA 需要复制的次数
    c_ceil = int(np.ceil(cc / c))
    cfa = np.tile(pcfa, (r_ceil, c_ceil, 1)) # 水平和垂直复制 CFA，填满整个图片

    image = np.pad(image, ((cfa.shape[0]-rr, 0), (cfa.shape[1]-cc, 0), (0, 0)), 'edge') # 扩展原图像使边缘缺损的CFA补全

    mosaickedImage = image[:, :, 0] * cfa[:, :, 0] + image[:, :, 1] * cfa[:, :, 1] + image[:, :, 2] * cfa[:, :, 2]

    return mosaickedImage

def mosaicking1(image):
    pcfa = np.zeros((8, 8, 3)) # 设置单个彩色滤波阵列
    pcfa[:, :, 0] = [[1, 0, 1, 0, 1, 0, 1, 0], # R通道
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],]
    pcfa[:, :, 1] = [[0, 1, 0, 1, 0, 1, 0, 1], # G通道
                    [1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0, 1, 0],]
    pcfa[:, :, 2] = [[0, 0, 0, 0, 0, 0, 0, 0], # B通道
                    [0, 1, 0, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 1, 0, 1], ]

    r = pcfa.shape[0] # 单个 CFA 尺寸
    c = pcfa.shape[1]
    rr = image.shape[0] # 图片尺寸
    cc = image.shape[1]
    r_ceil = int(np.ceil(rr / r)) # CFA 需要复制的次数
    c_ceil = int(np.ceil(cc / c))
    cfa = np.tile(pcfa, (r_ceil, c_ceil, 1)) # 水平和垂直复制 CFA，填满整个图片

    image = np.pad(image, ((cfa.shape[0]-rr, 0), (cfa.shape[1]-cc, 0), (0, 0)), 'edge') # 扩展原图像使边缘缺损的CFA补全

    mosaickedImage = image[:, :, 0] * cfa[:, :, 0] + image[:, :, 1] * cfa[:, :, 1] + image[:, :, 2] * cfa[:, :, 2]

    return mosaickedImage

# 下采样
def downSampling(raw, method):
    r, c = raw.shape
    half_r = r // 2
    half_c = c // 2
    downSampledImage = np.zeros((half_r, half_c))
    for row in range(half_r):
        for col in range(half_c):
            if method == "Mean":
                downSampledImage[row, col] = (raw[row * 2, col * 2] + raw[row * 2 + 1, col * 2 + 1]) / 2
            elif method == "Max":
                downSampledImage[row, col] = max(raw[row * 2, col * 2], raw[row * 2 + 1, col * 2 + 1])
            elif method == "Min":
                downSampledImage[row, col] = min(raw[row * 2, col * 2], raw[row * 2 + 1, col * 2 + 1])
            else:
                print("Error")

    return downSampledImage

# 重建马赛克，最简单方法：重新排列组合
def remosaicking(downSampledImage):
    bayerImage1 = np.zeros((downSampledImage.shape))
    r = downSampledImage.shape[0] // 4
    c = downSampledImage.shape[1] // 4
    for row in range(r):
        for col in range(c):
            block1 = downSampledImage[row*4:row*4+4, col*4:col*4+4]
            block2 = bayerImage1[row*4:row*4+4, col*4:col*4+4]
            # 重新排列 R
            block2[0, 0] = block1[0, 0]
            block2[0, 2] = block1[0, 1]
            block2[2, 0] = block1[1, 0]
            block2[2, 2] = block1[1, 1]
            # 重新排列 B
            block2[3, 3] = block1[3, 3]
            block2[1, 3] = block1[2, 3]
            block2[3, 1] = block1[3, 2]
            block2[1, 1] = block1[2, 2]
            # 重新排列 G
            block2[0, 3] = block1[0, 3]
            block2[0, 1] = block1[0, 2]
            block2[1, 2] = block1[1, 2]
            block2[2, 3] = block1[1, 3]
            block2[3, 0] = block1[3, 0]
            block2[1, 0] = block1[2, 0]
            block2[2, 1] = block1[2, 1]
            block2[3, 2] = block1[3, 1]

    return bayerImage1

# 上采样，最简单方法：复制
def upSampling(bayerImage1):
    bayerImage2 = np.zeros((bayerImage1.shape[0]*2, bayerImage1.shape[1]*2))
    r = bayerImage1.shape[0] // 4
    c = bayerImage1.shape[1] // 4
    for row in range(r):
        for col in range(c):
            block1 = bayerImage1[row*4:row*4+4, col*4:col*4+4]
            block2 = bayerImage2[row*8:row*8+8, col*8:col*8+8]
            block2[0:4, 0:4] = block2[0:4, 4:8] = block2[4:8, 0:4] = block2[4:8, 4:8] = block1

    return bayerImage2

def getNoisy(image):
    mean = 0
    sigma = 0.01
    # 生成高斯噪声
    gauss = np.random.normal(mean, sigma, image.shape)
    # 添加高斯噪声
    noisy_img = image + gauss

    return noisy_img


# 单通道引导滤波
def guidedFilter_oneChannel(srcImage, guidedImage, mask, eps=0.001):
    rad = 5  # 滤波窗口大小

    # 计算 mask 归一化因子
    mask_sum = cv2.boxFilter(mask, -1, (rad, rad), normalize=False) + 1e-8  # 避免除零

    # 计算加权均值
    P_mean = cv2.boxFilter(srcImage * mask, -1, (rad, rad), normalize=False) / mask_sum
    I_mean = cv2.boxFilter(guidedImage * mask, -1, (rad, rad), normalize=False) / mask_sum

    # 计算方差和协方差
    I_square_mean = cv2.boxFilter((guidedImage ** 2) * mask, -1, (rad, rad), normalize=False) / mask_sum
    I_mul_P_mean = cv2.boxFilter((srcImage * guidedImage) * mask, -1, (rad, rad), normalize=False) / mask_sum

    var_I = I_square_mean - I_mean * I_mean
    cov_I_P = I_mul_P_mean - I_mean * P_mean

    # 计算 a, b
    a = cov_I_P / (var_I + eps)
    b = P_mean - a * I_mean

    # 对 a, b 进行加权均值平滑
    a_mean = cv2.boxFilter(a * mask, -1, (rad, rad), normalize=False) / mask_sum
    b_mean = cv2.boxFilter(b * mask, -1, (rad, rad), normalize=False) / mask_sum

    # 计算最终输出
    dstImg = a_mean * guidedImage + b_mean

    return dstImg

def guidedFilter_threeChannel(bayerImage, guidedImage):
    r, c = bayerImage.shape

    srcImage = np.zeros((r, c, 3)) # 将单通道 bayer 扩展为三通道，像素相对位置不变，空缺补零
    mask = np.zeros((r, c, 3)) # 掩码，标记每一层哪些像素参与计算

    # 扩展 bayer，并计算掩码矩阵
    for row in range(r//2):
        for col in range(c//2):
            srcImage[row * 2, col * 2, 0] = bayerImage[row * 2, col * 2]
            mask[row * 2, col * 2, 0] = 1

            srcImage[row * 2, col * 2 + 1, 1] = bayerImage[row * 2, col * 2 + 1]
            mask[row * 2, col * 2 + 1, 1] = 1

            srcImage[row * 2 + 1, col * 2, 1] = bayerImage[row * 2 + 1, col * 2]
            mask[row * 2 + 1, col * 2, 1] = 1

            srcImage[row * 2 + 1, col * 2 + 1, 2] = bayerImage[row * 2 + 1, col * 2 + 1]
            mask[row * 2 + 1, col * 2 + 1, 2] = 1

    outputImage = np.zeros((r, c, 3)) # 初始化输出图像

    for dim in range(3):
        outputImage[:, :, dim] = guidedFilter_oneChannel(srcImage[:, :, dim], guidedImage, mask[:, :, dim])     #每一层都进行引导滤波

    # 注意！！白色像素作为引导图只有一个通道，将参与三次滤波计算，分别计算出三层的系数

    return outputImage

# 读入 rgbw 四通道图像
inputImage = scipy.io.loadmat('balloons_ms.mat')
inputImage = inputImage['rgbw']

# 提取 rgb 通道和 w 通道，并进行归一化，否则 bug
rgb = inputImage[:, :, :3] / 255
w = inputImage[:, :, 3] / 255

raw = mosaicking(rgb) # 马赛克化

# raw = getNoisy(raw)

quadBayer = downSampling(raw, 'Mean') # 下采样

bayerImage1 = remosaicking(quadBayer) # 重马赛克

bayerImage2 = upSampling(bayerImage1) # 上采样

outputImage = guidedFilter_threeChannel(bayerImage2, w) # 引导滤波去马赛克

# outputImage = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(bayerImage2) # 双线性插值去马赛克

outputImage = outputImage[0:inputImage.shape[0], 0:inputImage.shape[1]] # 裁剪输出图像，使其与输入图像尺寸相同，以便计算 PSNR

input_image = img_as_float(rgb)
output_image = img_as_float(outputImage)
psnr_value = compare_psnr(outputImage, rgb, data_range=1.0)
print(f"PSNR value: {psnr_value} dB")

#--------------------------------------------------------------------------------

rggb = mosaicking1(rgb)

# rggb = getNoisy(rggb)

# outputImage1 = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(rggb)

outputImage1 = guidedFilter_threeChannel(rggb, w)

outputImage1 = outputImage1[0:inputImage.shape[0], 0:inputImage.shape[1]]

input_image = img_as_float(rgb)
output_image = img_as_float(outputImage1)
psnr_value = compare_psnr(outputImage1, rgb, data_range=1.0)
print(f"PSNR value: {psnr_value} dB")

# 显示以上所有图像
fig, axes = plt.subplots(3, 3, figsize=(15, 10))

axes[0, 0].imshow(rgb)
axes[0, 0].set_title('输入图像的RGB通道')
axes[0, 0].axis('off')

axes[0, 1].imshow(raw, cmap='grey')
axes[0, 1].set_title('RAW图像')
axes[0, 1].axis('off')

axes[0, 2].imshow(quadBayer, cmap='grey')
axes[0, 2].set_title('四倍拜尔图像')
axes[0, 2].axis('off')

axes[1, 0].imshow(bayerImage1, cmap='grey')
axes[1, 0].set_title('拜尔图像')
axes[1, 0].axis('off')

axes[1, 1].imshow(bayerImage2, cmap='grey')
axes[1, 1].set_title('上采样后的拜尔图像')
axes[1, 1].axis('off')

axes[1, 2].imshow(outputImage)
axes[1, 2].set_title('输出图像')
axes[1, 2].axis('off')

axes[2, 0].imshow(outputImage1)
axes[2, 0].set_title('不经过上下采样，只进行引导滤波的输出图像')
axes[2, 0].axis('off')

plt.tight_layout()
plt.show()
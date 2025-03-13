import numpy as np
from extract_sub import  extract_sub
from guidedfilter import  guidedfilter
from construct_rgb import  construct_rgb
from interp_bilinear import interp_bilinearR, interp_bilinearG, interp_bilinearB

# 引导滤波去马赛克
def demosaick_rgb(raw, W):

    #提取R,G,B子图
    RGGB, WS = extract_sub(raw, W)

    #求得子图的滤波系数
    mean_a, mean_b = guidedfilter_joint(WS, RGGB)

    #将子图扩大回原图
    mean_a_full, mean_b_full = construct_rgb(mean_a, mean_b)

    #对原图中缺失的滤波系数进行双线性差值
    mean_a_full[:, :, 0] = interp_bilinearR(mean_a_full[:, :, 0])
    mean_a_full[:, :, 1] = interp_bilinearG(mean_a_full[:, :, 1])
    mean_a_full[:, :, 2] = interp_bilinearB(mean_a_full[:, :, 2])
    mean_b_full[:, :, 0] = interp_bilinearR(mean_b_full[:, :, 0])
    mean_b_full[:, :, 1] = interp_bilinearG(mean_b_full[:, :, 1])
    mean_b_full[:, :, 2] = interp_bilinearB(mean_b_full[:, :, 2])

    #计算输出图像
    outputImage = np.zeros(mean_a_full.shape)
    for i in range(outputImage.shape[2]):
        outputImage[:, :, i] = W[:, :] * mean_a_full[:, :, i] + mean_b_full[:, :, i]
    return outputImage

def guidedfilter_joint(WS, RGGB):
    mean_a = np.zeros(RGGB.shape)
    mean_b = np.zeros(RGGB.shape)
    for dim in range(RGGB.shape[2]):
        mean_a[:, :, dim], mean_b[:, :, dim] = guidedfilter(WS[:, :, dim], RGGB[:, :, dim])
    return mean_a, mean_b
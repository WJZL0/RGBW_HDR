import numpy as np
import cv2

def gauss_filter(r):
    r_max = max(r)
    ker = np.zeros((2 * r_max + 1, 2 * r_max + 1, len(r)))
    for i in range(len(r)):
        k = 2 * r[i] + 1
        this_ker = fspecial(k, (k + 1) / 6)
        s = r_max - r[i]
        ker[s:s+k, s:s+k, i] = this_ker
    return ker

def fspecial(k, sigma):
    return np.multiply(cv2.getGaussianKernel(k, sigma), (cv2.getGaussianKernel(k, sigma)).T)

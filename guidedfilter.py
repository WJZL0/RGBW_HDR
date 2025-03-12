import numpy as np
import cv2
from gauss_filter import gauss_filter

def guidedfilter(I, p):
    r = [1, 2, 3, 4]
    ker = gauss_filter(r)
    dim = ker.shape[2]
    a = np.zeros((I.shape[0], I.shape[1], dim))
    b = np.zeros((I.shape[0], I.shape[1], dim))
    res = np.zeros((I.shape[0], I.shape[1], dim))
    mean_a = np.zeros(I.shape)
    mean_b = np.zeros(I.shape)
    N_res = np.zeros(I.shape)


    for i in range(dim):
        a[:, :, i], b[:, :, i], res[:, :, i] = guided_filter(I, p, ker[:, :, i])
        N_res = N_res + cv2.filter2D(np.ones(I.shape), -1, ker[:, :, i])
        mean_a = mean_a + cv2.filter2D(a[:, :, i], -1, ker[:, :, i])
        mean_b = mean_b + cv2.filter2D(b[:, :, i], -1, ker[:, :, i])
    mean_a = mean_a / N_res
    mean_b = mean_b / N_res

    return mean_a, mean_b


def guided_filter(I, p, ker):
    N = cv2.filter2D(np.ones(I.shape), -1, ker)
    mean_I = cv2.filter2D(I, -1, ker) / N
    mean_II = cv2.filter2D(I * I, -1, ker) / N
    var_I = mean_II - mean_I * mean_I
    var_I = np.where(var_I < 1, 1, var_I)

    mean_p = cv2.filter2D(p, -1, ker) / N
    mean_Ip = cv2.filter2D(I * p, -1, ker) / N
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip / var_I
    b = mean_p - a * mean_I

    mean_pp = cv2.filter2D(p * p, -1, ker) / N
    res = a ** 2 * mean_II + mean_pp - 2 * a * mean_Ip + b ** 2 + 2 * b * a * mean_I - 2 * b * mean_p
    res = np.where(res < 1, 1, res)

    return a, b, res
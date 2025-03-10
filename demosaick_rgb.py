import numpy as np
import extract_sub
import guidedfilter
import construct_rgb

def demosaick_rgb(raw, W):
    RGGB, WS = extract_sub(raw, W)
    mean_a, mean_b = guidedfilter_joint(WS, RGGB)

    mean_a_full, mean_b_full = construct_rgb(mean_a, mean_b)

def guidedfilter_joint(WS, RGGB):
    mean_a = np.zeros(RGGB.shape)
    mean_b = np.zeros(RGGB.shape)
    for dim in range(RGGB.shape[2]):
        mean_a[:, :, dim], mean_b[:, : ,dim] = guidedfilter(WS[:, :, dim], RGGB[:, :, dim])
    return mean_a, mean_b
import numpy as np

def downSamplingRGB(raw, method):
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
                break
    return downSampledImage

def downSamplingW(W, method):
    r, c= W.shape
    half_r = r // 2
    half_c = c // 2
    downSampledImage = np.zeros((half_r, half_c))
    for row in range(half_r):
        for col in range(half_c):
            if method == "Mean":
                downSampledImage[row, col] = (W[row * 2, col * 2 + 1] + W[row * 2 + 1, col * 2]) / 2
            elif method == "Max":
                downSampledImage[row, col] = max(W[row * 2, col * 2 + 1] + W[row * 2 + 1, col * 2])
            elif method == "Min":
                downSampledImage[row, col] = min(W[row * 2, col * 2 + 1] + W[row * 2 + 1, col * 2])
            else:
                print("Error")
                break
    return downSampledImage
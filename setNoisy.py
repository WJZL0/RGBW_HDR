import numpy as np

def setNoisy(image):
    mean = 0
    sigma = 0.01
    # 生成高斯噪声
    gauss = np.random.normal(mean, sigma, image.shape)
    # 添加高斯噪声
    noisy_img = image + gauss

    return noisy_img
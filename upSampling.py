import numpy as np

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
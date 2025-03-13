import numpy as np

def construct_rgb(mean_a, mean_b):
    rr = mean_a.shape[0]
    cc = mean_a.shape[1]
    row = rr * 2
    col = cc * 2
    mean_a_full = np.zeros((row, col, 3))
    mean_b_full = np.zeros((row, col, 3))

    #R
    for i in range(rr // 2):
        for j in range(cc // 2):
            mean_a_full[i * 4, j * 4, 0] = mean_a[i * 2, j * 2, 0]
            mean_a_full[i * 4, j * 4 + 1, 0] = mean_a[i * 2, j * 2 + 1, 0]
            mean_a_full[i * 4 + 1, j * 4, 0] = mean_a[i * 2 + 1, j * 2, 0]
            mean_a_full[i * 4 + 1, j * 4 + 1, 0] = mean_a[i * 2 + 1, j * 2 + 1, 0]

            mean_b_full[i * 4, j * 4, 0] = mean_b[i * 2, j * 2, 0]
            mean_b_full[i * 4, j * 4 + 1, 0] = mean_b[i * 2, j * 2 + 1, 0]
            mean_b_full[i * 4 + 1, j * 4, 0] = mean_b[i * 2 + 1, j * 2, 0]
            mean_b_full[i * 4 + 1, j * 4 + 1, 0] = mean_b[i * 2 + 1, j * 2 + 1, 0]

    #G
    for i in range(rr // 2):
        for j in range(cc // 2):
            mean_a_full[i * 4, j * 4 + 2, 1] = mean_a[i * 2, j * 2, 1]
            mean_a_full[i * 4, j * 4 + 2 + 1, 1] = mean_a[i * 2, j * 2 + 1, 1]
            mean_a_full[i * 4 + 1, j * 4 + 2, 1] = mean_a[i * 2 + 1, j * 2, 1]
            mean_a_full[i * 4 + 1, j * 4 + 2 + 1, 1] = mean_a[i * 2 + 1, j * 2 + 1, 1]

            mean_a_full[i * 4 + 2, j * 4, 1] = mean_a[i * 2, j * 2, 2]
            mean_a_full[i * 4 + 2, j * 4 + 1, 1] = mean_a[i * 2, j * 2 + 1, 2]
            mean_a_full[i * 4 + 2 + 1, j * 4, 1] = mean_a[i * 2 + 1, j * 2, 2]
            mean_a_full[i * 4 + 2 + 1, j * 4 + 1, 1] = mean_a[i * 2 + 1, j * 2 + 1, 2]

            mean_b_full[i * 4, j * 4 + 2, 1] = mean_b[i * 2, j * 2, 1]
            mean_b_full[i * 4, j * 4 + 2 + 1, 1] = mean_b[i * 2, j * 2 + 1, 1]
            mean_b_full[i * 4 + 1, j * 4 + 2, 1] = mean_b[i * 2 + 1, j * 2, 1]
            mean_b_full[i * 4 + 1, j * 4 + 2 + 1, 1] = mean_b[i * 2 + 1, j * 2 + 1, 1]

            mean_b_full[i * 4 + 2, j * 4, 1] = mean_b[i * 2, j * 2, 2]
            mean_b_full[i * 4 + 2, j * 4 + 1, 1] = mean_b[i * 2, j * 2 + 1, 2]
            mean_b_full[i * 4 + 2 + 1, j * 4, 1] = mean_b[i * 2 + 1, j * 2, 2]
            mean_b_full[i * 4 + 2 + 1, j * 4 + 1, 1] = mean_b[i * 2 + 1, j * 2 + 1, 2]

    #B
    for i in range(rr // 2):
        for j in range(cc // 2):
            mean_a_full[i * 4 + 2, j * 4 + 2, 2] = mean_a[i * 2, j * 2, 3]
            mean_a_full[i * 4 + 2, j * 4 + 2 + 1, 2] = mean_a[i * 2, j * 2 + 1, 3]
            mean_a_full[i * 4 + 2 + 1, j * 4 + 2, 2] = mean_a[i * 2 + 1, j * 2, 3]
            mean_a_full[i * 4 + 2 + 1, j * 4 + 2 + 1, 2] = mean_a[i * 2 + 1, j * 2 + 1, 3]

            mean_b_full[i * 4 + 2, j * 4 + 2, 2] = mean_b[i * 2, j * 2, 3]
            mean_b_full[i * 4 + 2, j * 4 + 2 + 1, 2] = mean_b[i * 2, j * 2 + 1, 3]
            mean_b_full[i * 4 + 2 + 1, j * 4 + 2, 2] = mean_b[i * 2 + 1, j * 2, 3]
            mean_b_full[i * 4 + 2 + 1, j * 4 + 2 + 1, 2] = mean_b[i * 2 + 1, j * 2 + 1, 3]

    return mean_a_full, mean_b_full
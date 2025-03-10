import numpy as np

def extract_sub(raw, W):
        row, col = raw.shape
        rr = row // 2
        cc = col // 2
        RGGB = np.zeros((rr, cc, 4))
        WS = np.zeros((rr, cc, 4))

        #R
        for i in range(rr // 2):
            for j in range(cc // 2):
                RGGB[i * 2, j * 2, 0] = raw[i * 4, j * 4]
                RGGB[i * 2, j * 2 + 1, 0] = raw[i * 4, j * 4 + 1]
                RGGB[i * 2 + 1, j * 2, 0] = raw[i * 4 + 1, j * 4]
                RGGB[i * 2 + 1, j * 2 + 1, 0] = raw[i * 4 + 1, j * 4 + 1]
                WS[i * 2, j * 2, 0] = W[i * 4, j * 4]
                WS[i * 2, j * 2 + 1, 0] = W[i * 4, j * 4 + 1]
                WS[i * 2 + 1, j * 2, 0] = W[i * 4 + 1, j * 4]
                WS[i * 2 + 1, j * 2 + 1, 0] = W[i * 4 + 1, j * 4 + 1]
        #G1
        for i in range(rr // 2):
            for j in range(cc // 2):
                RGGB[i * 2, j * 2, 1] = raw[i * 4, j * 4 + 2]
                RGGB[i * 2, j * 2 + 1, 1] = raw[i * 4, j * 4 + 2 + 1]
                RGGB[i * 2 + 1, j * 2, 1] = raw[i * 4 + 1, j * 4 + 2]
                RGGB[i * 2 + 1, j * 2 + 1, 1] = raw[i * 4 + 1, j * 4 + 2 + 1]
                WS[i * 2, j * 2, 1] = W[i * 4, j * 4 + 2]
                WS[i * 2, j * 2 + 1, 1] = W[i * 4, j * 4 + 2 + 1]
                WS[i * 2 + 1, j * 2, 1] = W[i * 4 + 1, j * 4 + 2]
                WS[i * 2 + 1, j * 2 + 1, 1] = W[i * 4 + 1, j * 4 + 2 + 1]
        #G2
        for i in range(rr // 2):
            for j in range(cc // 2):
                RGGB[i * 2, j * 2, 2] = raw[i * 4 + 2, j * 4]
                RGGB[i * 2, j * 2 + 1, 2] = raw[i * 4 + 2, j * 4 + 1]
                RGGB[i * 2 + 1, j * 2, 2] = raw[i * 4 + 2 + 1, j * 4]
                RGGB[i * 2 + 1, j * 2 + 1, 2] = raw[i * 4 + 2 + 1, j * 4 + 1]
                WS[i * 2, j * 2, 2] = W[i * 4 + 2, j * 4]
                WS[i * 2, j * 2 + 1, 2] = W[i * 4 + 2, j * 4 + 1]
                WS[i * 2 + 1, j * 2, 2] = W[i * 4 + 2 + 1, j * 4]
                WS[i * 2 + 1, j * 2 + 1, 2] = W[i * 4 + 2 + 1, j * 4 + 1]
        #B
        for i in range(rr // 2):
            for j in range(cc // 2):
                RGGB[i * 2, j * 2, 3] = raw[i * 4 + 2, j * 4 + 2]
                RGGB[i * 2, j * 2 + 1, 3] = raw[i * 4 + 2, j * 4 + 2 + 1]
                RGGB[i * 2 + 1, j * 2, 3] = raw[i * 4 + 2 + 1, j * 4 + 2]
                RGGB[i * 2 + 1, j * 2 + 1, 3] = raw[i * 4 + 2 + 1, j * 4 + 2 + 1]
                WS[i * 2, j * 2, 3] = W[i * 4 + 2, j * 4 + 2]
                WS[i * 2, j * 2 + 1, 3] = W[i * 4 + 2, j * 4 + 2 + 1]
                WS[i * 2 + 1, j * 2, 3] = W[i * 4 + 2 + 1, j * 4 + 2]
                WS[i * 2 + 1, j * 2 + 1, 3] = W[i * 4 + 2 + 1, j * 4 + 2 + 1]

        return RGGB, WS
import numpy as np

def arrange(im, survival_rate):
    size = np.shape(im) # [64, 28, 28, 1]
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                for l in range(size[3]):
                    if im[i][j][k][l] < 0.01 or np.random.rand(1) > survival_rate:
                        im[i][j][k][l] = 0.
    return im
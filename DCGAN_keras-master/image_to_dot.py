import numpy as np

def arrange(im):
    size = np.shape(im) # [64, 28, 28, 1]
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                for l in range(size[3]):
                    if im[i][j][k][l] > 0.5 and np.random.rand(1) > 0.001:
                        im[i][j][k][l] = 0.
    return im
import numpy as np

np.set_printoptions(threshold=np.inf)


# 行列変換ライブラリとして使用
def grid_points(shape, dtype=np.float32):
    """Return the grid points of a given shape with distance `1`."""

    """
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    y = y.flatten()
    x = x.flatten()

    z = np.empty((shape[0] * shape[1], 2), dtype)
    z[:, 0] = y
    z[:, 1] = x
    return z
    """
    z = []
    for i in range(shape[0]):
        for j in range(shape[1] - 1):
            z.append((i * 32 + j, i * 32 + j + 1))
    for i in range(shape[0] - 1):
        for j in range(shape[1]):
            z.append((i * 32 + j, (i + 1) * 32 + j))
    return z

# grid = grid_points((32, 32))
# print(grid)

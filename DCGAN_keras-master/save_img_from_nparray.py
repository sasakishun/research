import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from datetime import datetime
import cv2


class SaveImgFromList:
    def __init__(self, imgs, shape, tag=None, comment=None):
        self.imgs = [np.array(img) * 255. for img in imgs]
        self.shape = shape
        self.tag = tag
        if comment is not None:
            comment = "_" + comment
        self.path = os.getcwd() + r"\corrected_img\{}.jpg".format(datetime.now().strftime("%Y%m%d%H%M%S") + comment)

    def __call__(self, *args, **kwargs):
        if self.shape[0] == 1 or self.shape[1] == 1:
            return
        imgs = [Image.fromarray(np.reshape(img, self.shape)).convert('RGB') for img in self.imgs]
        fig = plt.figure(figsize=(6, 8 * len(imgs)))
        print(imgs)
        for i, p in enumerate(imgs):
            ax = fig.add_subplot(1, len(imgs) + 1, i + 1)
            ax.imshow(imgs[i])  # , interpolation=p)
            ax.set_axis_off()
            ax.set_title(self.tag[i])

        # 一番最初の画像と修正後の画像の変更箇所を明示
        prev = self.imgs[0]
        corrected = self.imgs[-1]
        ax = fig.add_subplot(1, len(imgs) + 1, len(imgs) + 1)
        ax.imshow(Image.fromarray(colorize_pixel_diff(corrected, prev, self.shape)))
        ax.set_axis_off()
        ax.set_title("diff")
        plt.savefig(self.path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print("saved to -> {}".format(self.path))


def colorize_pixel_diff(corrected, prev, shape):
    _diff = np.subtract(corrected, prev)
    diff = np.zeros((shape[0], shape[1], 3))
    for i in range(_diff.shape[0]):
        for j in range(_diff.shape[1]):
            if _diff[i, j] > 0:
                diff[i, j, 0] = _diff[i, j]
            elif _diff[i, j] < 0:
                diff[i, j, 2] = -_diff[i, j]
    # print("_diff:{}".format(_diff))
    # print("diff:{}".format(diff))
    return np.array(diff).astype(np.uint8)


import numpy as np

if __name__ == '__main__':
    shape = (8, 8)
    array = np.zeros(shape)
    array[1, 1] = 1
    corrected = np.zeros(shape)
    corrected[0, 0] = 1
    SaveImgFromList([array] + [array] + [array] + [corrected],
                    shape,
                    tag=["ori[{}]".format(1)] +
                        ["cor[{}]".format(2) for _ in range(4)],
                    comment="corrected_class[{}]_sample[{}]".format(0, 0))()

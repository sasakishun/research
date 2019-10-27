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
        self.path = os.getcwd() + r"\corrected_img\{}.jpg".format(datetime.now().strftime("%Y%m%d%H%M%S")+comment)

    def __call__(self, *args, **kwargs):
        if self.shape[0] == 1  or self.shape[1] == 1:
            return
        imgs = [Image.fromarray(np.reshape(img, self.shape)).convert('RGB') for img in self.imgs]
        fig = plt.figure(figsize=(6, 8*len(imgs)))
        print(imgs)
        for i, p in enumerate(imgs):
            ax = fig.add_subplot(1, len(imgs)+1, i+1)
            ax.imshow(imgs[i])# , interpolation=p)
            ax.set_axis_off()
            ax.set_title(self.tag[i])

        # 一番最初の画像と修正後の画像の変更箇所を明示
        prev = self.imgs[0]
        corrected = self.imgs[-1]
        diff = np.abs(np.subtract(corrected,  prev))
        ax = fig.add_subplot(1, len(imgs)+1, len(imgs) + 1)
        ax.imshow(Image.fromarray(np.reshape(diff, self.shape)).convert('RGB'))
        ax.set_axis_off()
        ax.set_title("diff")
        plt.savefig(self.path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print("saved to -> {}".format(self.path))

import numpy as np
if __name__ == '__main__':
    array = np.ones((28, 28))
    SaveImgFromList([array] + [array] + [array] + [array],
                    (28, 28),
                    tag=["original[{}]".format(1)] +
                        ["corrected[{}]".format(2) for _ in range(4)],
                    comment="corrected_class[{}]_sample[{}]".format(0, 0))()

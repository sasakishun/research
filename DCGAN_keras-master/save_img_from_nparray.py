import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from datetime import datetime
import cv2


class SaveImgFromList:
    def __init__(self, imgs, shape, tag=None, comment=None, dir=None, output=None):
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})  # 桁を揃える
        self.imgs = [np.reshape(np.array(img), shape) for img in [imgs[0], imgs[-1]]]
        self.shape = shape
        self.tag = tag
        if comment is not None:
            comment = "_" + comment
        self.path = os.getcwd()
        if dir is not None:
            self.path += r"\visualized_iris\hidden_output" + dir
            from visualization import my_makedirs
            my_makedirs(self.path)
            self.path += r"\{}.jpg".format(
                "ex2_{}".format(tag[0][1]))  # datetime.now().strftime("%Y%m%d%H%M%S") + comment)
        else:
            self.path += r"\corrected_img\{}.jpg".format(datetime.now().strftime("%Y%m%d%H%M%S")
                                                         + (comment if comment is not None else ""))
        self.output = output

    def __call__(self, *args, **kwargs):
        # テキストファイルとして保存
        # print("self.output:{}".format(self.output))
        from binary__tree_main import write_result
        result = [str(self.imgs[0])]
        if self.output is not None:
            result.append("-> " + str(self.output[0][0][0]) + " -> " + self.tag[0])
        result.append("corrected ->")
        for i, img in enumerate(self.imgs):
            if i == 0:
                continue
            result.append(str(img))
            if self.output is not None:
                result.append("-> " + str(self.output[i][0][0]) + " -> " + self.tag[i])
        write_result(self.path[:-4] + ".txt", result)
        if self.shape[0] == 1 or self.shape[1] == 1:
            return
        self.imgs = [(np.ones(np.shape(img)) - np.array(img)) * 255. for img in self.imgs]
        # for i in self.imgs:
            # print("imgs\n{}".format(i))
        imgs = [Image.fromarray(img).convert('RGB') for img in self.imgs]
        fig = plt.figure(figsize=(6, 8 * len(imgs)))
        # print(imgs)
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
    # クラス初期化の際に色反転してるから_diff>0の部分が消えた箇所
    diff = np.ones((shape[0], shape[1], 3)) * 255
    for i in range(shape[0]):
        for j in range(shape[1]):
            # print("i:{} j:{}".format(i, j))
            exaggerate = 100
            # 消えたピクセル
            if _diff[i][j] >= 1:
                diff[i][j][0] = 255 - min(255., _diff[i][j] + exaggerate)
                diff[i][j][1] = 255 - min(255., _diff[i][j] + exaggerate)
                diff[i][j][2] = 255
            # 新たに出現したピクセル
            elif _diff[i][j] <= -1:
                diff[i][j][0] = 255
                diff[i][j][1] = 255 + max(-255., _diff[i][j] - exaggerate)
                diff[i][j][2] = 255 + max(-255., _diff[i][j] - exaggerate)
            else:
                diff[i][j][0] = prev[i][j]
                diff[i][j][1] = prev[i][j]
                diff[i][j][2] = prev[i][j]
    # print("_diff:{}".format(_diff))
    # print("diff:{}".format(diff))
    return np.array(diff).astype(np.uint8)


import numpy as np

if __name__ == '__main__':
    shape = (4, 4)
    array = np.zeros(shape)
    array[1, 1] = 255
    array[2, 2] = 255
    array[3, 3] = 255
    corrected = np.zeros(shape)
    corrected[1, 1] = 100
    corrected[2, 2] = 100
    corrected[0, 3] = 100
    corrected[0, 1] = 100
    SaveImgFromList([array] + [corrected],
                    shape,
                    tag=["ori[{}]".format(1)] +
                        ["cor[{}]".format(2) for _ in range(1)],
                    comment="corrected_class[{}]_sample[{}]".format(0, 0))()

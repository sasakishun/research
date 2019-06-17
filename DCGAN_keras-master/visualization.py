import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick  # 目盛り操作に必要なライブラリを読み込みます
import pylab
from datetime import datetime
import config_mnist as cf
import cv2


def visualize(x, y, labels, ite, testflag, showflag=False):
    plt.figure(figsize=(10, len(x[0][0]) // 2 + 5), dpi=100)
    # colors = ["tomato", "black", "lightgreen"]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [colors[0],
              colors[8],
              colors[3],
              colors[1],
              colors[2],
              colors[4],
              colors[5],
              colors[6],
              colors[7],
              colors[9]]
    # プロット
    for i in range(len(x)):
        for j in range(len(x[i][0])):
            if j == 0:
                plt.scatter([j + 0.025 * i - 0.025 for _ in range(len(x[i]))], np.array(x[i])[:, j], color=colors[i],
                            s=5, label=i)
                plt.legend(loc='uppper right',
                           bbox_to_anchor=(0.75, 0.5, 0.5, .100),
                           # borderaxespad=0.,
                           facecolor="white")  # colors[i])
                # plt.legend(loc='lower right', facecolor=colors[i])
            else:
                plt.scatter([j + 0.025 * i - 0.025 for _ in range(len(x[i]))], np.array(x[i])[:, j], color=colors[i],
                            s=5)
    plt.title("ite:{} {}".format(ite, "test" if testflag else "train"))
    plt.xlabel("hidden node")
    plt.ylabel("output")
    plt.xticks(range(len(x[i][0])))

    # y軸に1刻みにで小目盛り(minor locator)表示
    plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(1))
    # 小目盛りに対してグリッド表示
    plt.grid(which='minor')
    # 右側の余白を調整
    pylab.subplots_adjust(right=0.7)

    # save as png
    path = r"C:\Users\papap\Documents\research\DCGAN_keras-master\visualized_iris"
    print("saved to -> " + path + "\{}{}".format("test" if testflag else "train", ite))
    if ite % cf.Iteration == 0:
        plt.savefig(path + "{}{}_{}".format(r"\test\test" if testflag else r"\train\train", ite,
                                            datetime.now().strftime("%Y%m%d%H%M%S")))
    else:
        plt.savefig(path + "{}{}".format(r"\test\test" if testflag else r"\train\train", ite))
    if showflag:
        path = r"C:\Users\papap\Documents\research\DCGAN_keras-master\visualized_iris\ネットワークアーキテクチャ"
        path += "{}{}_{}".format(
            r"\test" if testflag else r"\train",
            r"\{}".format(datetime.now().strftime("%Y%m%d%H%M%S")),
            "test" if testflag else "train")
        plt.savefig(path)
        # plt.show()
    plt.close()
    return cv2.imread(path + ".pmg")


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)



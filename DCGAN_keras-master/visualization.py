import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick # 目盛り操作に必要なライブラリを読み込みます
import pylab
from datetime import datetime
import config_mnist as cf


def visualize(x, y, labels, ite, testflag, showflag=False):
    plt.figure(figsize=(10, len(x[0][0])//2 + 5), dpi=100)
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
                plt.scatter(np.array(x[i])[:,j], [j+0.025*i - 0.025 for _ in range(len(x[i]))], color=colors[i], s=5, label=i)
                plt.legend(loc='uppper right',
                           bbox_to_anchor=(0.75, 0.5, 0.5, .100),
                           # borderaxespad=0.,
                           facecolor = "white")# colors[i])
                # plt.legend(loc='lower right', facecolor=colors[i])
            else:
                plt.scatter(np.array(x[i])[:, j], [j + 0.025 * i - 0.025 for _ in range(len(x[i]))], color=colors[i], s=5)
    plt.title("ite:{} {}".format(ite, "test" if testflag else "train"))
    plt.xlabel("x axis")
    plt.yticks(range(len(x[i][0])))
    plt.ylabel("hidden node")

    # y軸に1刻みにで小目盛り(minor locator)表示
    plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(1))
    # 小目盛りに対してグリッド表示
    plt.grid(which='minor')
    # 右側の余白を調整
    pylab.subplots_adjust(right=0.7)

    # save as png
    path = r"C:\Users\papap\Documents\research\DCGAN_keras-master\visualized_iris"
    print("saved to -> " + path +"\{}{}".format("test" if testflag else "train", ite))
    if ite % cf.Iteration == 0:
        plt.savefig(path+"{}{}_{}".format(r"\test\test" if testflag else r"\train\train", ite, datetime.now().strftime("%Y%m%d%H%M%S")))
    else:
        plt.savefig(path + "{}{}".format(r"\test\test" if testflag else r"\train\train", ite))
    if showflag:
        path = r"C:\Users\papap\Documents\research\DCGAN_keras-master\visualized_iris\ネットワークアーキテクチャ"
        plt.savefig(path + "{}{}_{}".format(
            r"\test" if testflag else r"\train",
            r"\{}".format(datetime.now().strftime("%Y%m%d%H%M%S")),
            "test" if testflag else "train"))
        # plt.show()
    plt.close()
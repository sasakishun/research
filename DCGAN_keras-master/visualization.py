import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick # 目盛り操作に必要なライブラリを読み込みます
import pylab

def visualize(x, y, labels, ite, testflag):
    # colors = ["tomato", "black", "lightgreen"]
    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900',
              '#009999']
    # プロット
    for i in range(len(x)):
        for j in range(len(x[i][0])):
            if j == 0:
                plt.scatter(np.array(x[i])[:,j], [j*4+0.1*i - 0.1 for _ in range(len(x[i]))], color=colors[i], s=5, label=i)
                plt.legend(loc='uppper right',
                           bbox_to_anchor=(0.75, 0.5, 0.5, .100),
                           # borderaxespad=0.,
                           facecolor = colors[i])
                # plt.legend(loc='lower right', facecolor=colors[i])
            else:
                plt.scatter(np.array(x[i])[:, j], [j * 4 + 0.1 * i - 0.1 for _ in range(len(x[i]))], color=colors[i], s=5)
    plt.title("ite:{} {}".format(ite, "test" if testflag else "train"))
    plt.xlabel("x axis")
    plt.ylabel("hidden node")

    # y軸に1刻みにで小目盛り(minor locator)表示
    plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(2))
    # 小目盛りに対してグリッド表示
    plt.grid(which='minor')
    # 右側の余白を調整
    pylab.subplots_adjust(right=0.7)
    """
    plt.legend(loc='uppper right',
               bbox_to_anchor=(1.05, 0.5, 0.5, .100),
               borderaxespad=0., )
   """
    # plt.grid(True, axis="y")

    # plt.xlim(0, 10)
    # plt.ylim(0, 20)
    # plt.show()

    # save as png
    path = r"C:\Users\papap\Documents\research\DCGAN_keras-master\visualized_iris"
    print("saved to -> " + path +"\{}{}".format("test" if testflag else "train", ite))
    plt.savefig(path+"\{}{}".format("test" if testflag else "train", ite))
    plt.close()
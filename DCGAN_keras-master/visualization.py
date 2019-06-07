import numpy as np
import matplotlib.pyplot as plt

def visualize(x, y, ite, testflag):
    color = ["tomato", "black", "lightgreen"]
    # プロット
    for i in range(3):
        for j in range(len(x[i][0])):
            plt.scatter(np.array(x[i])[:,j], [j+0.1*i for _ in range(len(x[i]))], color=color[i], s=5)
    plt.title("ite:{} {}".format(ite, "test" if testflag else "train"))
    plt.xlabel("x axis")
    plt.ylabel("hidden node")
    plt.grid(True, axis="y")
    # plt.show()

    # save as png
    path = r"C:\Users\papap\Documents\research\DCGAN_keras-master\visualized_iris"
    print(path+"\{}{}".format("test" if testflag else "train", ite))
    plt.savefig(path+"\{}{}".format("test" if testflag else "train", ite))
    plt.close()
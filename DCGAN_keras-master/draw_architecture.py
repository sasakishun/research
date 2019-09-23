from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np

vertical_distance_between_layers = 60  # 6
horizontal_distance_between_neurons = 10  # 2
neuron_radius = 0.5
number_of_neurons_in_widest_layer = 4
from datetime import datetime
import cv2
import os

max_weight = 0


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, text="", color="black"):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False, color=color)
        # _text = pyplot.text(self.x-0.25, self.y-0.25, text, fontsize=neuron_radius*10)
        # pyplot.gca()._add_text(_text)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights, non_active_neurons=None):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights
        self.neuron_color = ["black" for _ in range(number_of_neurons)]
        if non_active_neurons:
            for i in non_active_neurons:
                non_active_neurons[i] = "white"

    def __intialise_neurons(self, number_of_neurons, image_flag = False):
        # image_flag == Trueだと入力:64を8*8にして可視化
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            if image_flag and number_of_neurons == 64:
                if iteration % 8 == 0:
                    x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons // 8)
                neuron = Neuron(x, self.y - 64 * (iteration // 8) - 40)
                neurons.append(neuron)
                # print("number of neurons:{}".format(number_of_neurons))
                x += horizontal_distance_between_neurons * (8 * 64 / number_of_neurons)
            else:
                neuron = Neuron(x, self.y)
                neurons.append(neuron)
                # print("number of neurons:{}".format(number_of_neurons))
                x += horizontal_distance_between_neurons * (64 / number_of_neurons)
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (16 / number_of_neurons) \
               * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        # line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        color = "red" if linewidth > 0 else "blue"
        linewidth = abs(linewidth)
        if linewidth > 0.:
            linewidth = max(0.2, linewidth)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=abs(linewidth), color=color, alpha=abs(linewidth))
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw(text=this_layer_neuron_index, color=self.neuron_color[this_layer_neuron_index])
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    # print("weight:{}".format(weight))
                    # print("alpha:{}".format(weight / np.max(abs(self.previous_layer.weights))))
                    # print("weight:{}".format(self.previous_layer.weights))
                    # print("max_weight:{}".format(np.max(abs(self.previous_layer.weights))))
                    self.__line_between_two_neurons(neuron, previous_layer_neuron,
                                                    linewidth=weight / np.max(abs(self.previous_layer.weights)))
                    # weightはスカラー値であり、重みの大きさ＝結合の太さ(linewidth)とする
                    # 正の重みは赤、負の重みは青で表示
                    # linewidthだと整数値しか扱えない -> 透明度で結合強度を表現した方がいい


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None, non_active_neurons=None):
        layer = Layer(self, number_of_neurons, weights, non_active_neurons)
        self.layers.append(layer)

    """
    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()
    """

    def draw(self, path, acc=-1, comment=""):
        for layer in self.layers:
            layer.draw()
        pyplot.title("{} acc:{:.4f}".format(comment, acc))
        pyplot.axis('scaled')
        pyplot.tick_params(labelbottom=False,
                           labelleft=False,
                           labelright=False,
                           labeltop=False)
        pyplot.xlabel("input")
        if not os.path.exists(path):  # ディレクトリがないとき新規作成
            path = r"C:\Users\xeno\Documents\research\DCGAN_keras-master\visualized_iris\network_architecture"
            print("path:{}".format(path))
        path += "{}{}_{}".format(r"\test", r"\{}".format(datetime.now().strftime("%Y%m%d%H%M%S")), "architecture")
        pyplot.savefig(path, bbox_inches="tight", pad_inches=0.0, dpi=200)
        print("saved to -> {}".format(path))
        # pyplot.show()
        pyplot.close()
        return cv2.imread(path + ".png")


def mydraw(_weights, acc, comment="", non_active_neurons=None):
    vertical_distance_between_layers = 6
    horizontal_distance_between_neurons = 2
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 4
    network = NeuralNetwork()
    # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)

    weights = []
    nodes = []
    # print("_weights:{}\n{}".format([np.shape(weight) for weight in _weights], _weights))
    for i in range(np.shape(_weights)[0]):
        # print("\n_weights[{}]:{}\n{}".format(i, np.shape(_weights[i]), _weights[i]))
        if _weights[i].ndim == 2:
            nodes.append(np.shape(_weights[i])[0])
            weights.append(_weights[i])
            # print("add weights")
            # print("weights[{}]:{}\n{}".format(i, np.shape(weights), weights))
            # for i in range(np.shape(_weights[0])[0]):
            # for j in range(np.shape(_weights[0])[1]):
    global max_weight
    for weight in weights:
        max_weight = max(max_weight, np.max(abs(weight)))
    print("max_weight:{}".format(max_weight))

    nodes.append(np.shape(weights[-1])[1])
    # print("weights:{}\n{}".format([np.shape(weight) for weight in weights], weights))
    # for i in range(len(nodes)-1):
    # weights.append(np.ones((nodes[i+1], nodes[i])))

    print("nodes of each layer:{}".format(nodes))
    print("weights:{}".format([np.shape(weights[i]) for i in range(len(weights))]))
    print("_weights:{}".format([np.shape(_weights[i]) for i in range(len(_weights))]))

    for i in range(len(nodes) - 1):
        network.add_layer(nodes[i], weights[i].T, non_active_neurons[i] if non_active_neurons else None)
    network.add_layer(nodes[-1])
    # print("weights:\n{}".format(weights))
    path = os.getcwd() + r"\visualized_iris\network_architecture"
    return network.draw(path=path, acc=acc, comment=comment)


def _active_route(_weights, acc, comment="", binary_target=-1, using_nodes=[]):
    # _weightsはバイアスを含まない重み集合
    import config_mnist as cf
    import copy
    weights = []
    for i in range(np.shape(_weights)[0]):
        # print("\n_weights[{}]:{}\n{}".format(i, np.shape(_weights[i]), _weights[i]))
        if _weights[i].ndim == 2:
            weights.append(_weights[i])
    print("\nbinary_target:{}".format(binary_target))
    print("weights")
    for i in range(len(weights)):
        print(weights[i].shape)
    print("using_nodes:{}".format(using_nodes))
    if binary_target >= 0:
        g_mask_1 = [[0 for _ in range(weights[0].shape[1])], [0 for _ in range(weights[1].shape[1])]]
        # g_mask_1 = [[今のactive], [次層のactive]]
        for i in range(using_nodes[0], sum(using_nodes)):
            g_mask_1[0][i] = 1

        for i in range(len(weights) - 1):
            print("g_mask_1:{}".format([len(g_mask_1[0]), len(g_mask_1[1])]))
            print("g_mask_1[0]:{}".format(g_mask_1[0]))
            print("g_mask_1[1]:{}".format(g_mask_1[1]))
            print("i:{} / {}".format(i, len(weights) - 1))
            # i層目に着目
            for j in range(weights[i].shape[0]):
                # i層目j番ノードに着目
                print(" j:{} / {}".format(j, weights[i].shape[0]))
                for k in range(weights[i][j].shape[0]):
                    if g_mask_1[0][k] == 0:
                        # i層目j番ノードがactiveでない->繋がる重みを全て0に
                        weights[i][j][k] = 0
                        # i+1層目k番ノードがactive->そのノードとi+2層と繋がるノードをactiveに
                        # if weights[i][j][k] != 0.:
                        # g_mask_1[1][k] = 1
            for k in range(weights[i + 1].shape[0]):
                for l in range(weights[i + 1][k].shape[0]):
                    if g_mask_1[0][k] == 0:
                        # i層目j番ノードがactiveでない->繋がる重みを全て0に
                        weights[i + 1][k][l] = 0
            # 次のactiveを設定
            if i < len(weights) - 1:
                for k in range(weights[i + 1].shape[0]):
                    if g_mask_1[0][k] == 1:
                        for l in range(weights[i + 1][k].shape[0]):
                            if weights[i + 1][k][l] != 0:
                                g_mask_1[1][l] = 1
            if i < len(weights) - 2:
                g_mask_1 = [copy.deepcopy(g_mask_1[1]), [0 for _ in range(weights[i + 2].shape[1])]]
            else:
                g_mask_1 = [g_mask_1[1], []]
    return mydraw(weights, acc, comment=comment + "\nusing_node:[{}:{}]".format(using_nodes[0], sum(using_nodes)))


def active_route(_weights, acc, comment="", binary_target=-1, using_nodes=[], active_nodes=[]):
    # _weights : バイアスを含まない重み行列
    # active_nodes : クラス(binary_target)において使用するノード番号リスト
    import config_mnist as cf
    import copy
    print("active_nodes:{}".format(active_nodes))
    weights = []
    for i in range(np.shape(_weights)[0]):
        if _weights[i].ndim == 2:
            weights.append(_weights[i])
    _active_nodes = [[0 for _ in range(np.shape(weights[i])[0])] for i in range(len(weights))]
    print("_active_nodes:{}".format([np.shape(i) for i in _active_nodes]))
    if not active_nodes[0]:
        _active_nodes[0] = [1 for _ in range(np.shape(_weights[0])[0])]
    for i in range(len(active_nodes)):
        for j in range(len(active_nodes[i])):
            _active_nodes[i][j] = 1
    for i in range(len(_active_nodes)):
        print("_active_nodes[{}]:{}".format(i, _active_nodes[i]))
    print("\nbinary_target:{}".format(binary_target))
    print("weights")
    for i in range(len(weights)):
        print(weights[i].shape)
    if binary_target >= 0:
        for i in range(1, len(weights)):
            # i層目に着目
            for j in range(weights[i].shape[0]):
                # i層目j番ノードに着目
                if _active_nodes[i][j] == 0:
                    for k in range(weights[i][j].shape[0]):
                        weights[i][j][k] = 0
                    for k in range(weights[i - 1].shape[0]):
                        weights[i - 1][k][j] = 0
    return mydraw(weights, acc, comment=comment + "\nusing_node:[{}:{}]".format(using_nodes[0], sum(using_nodes)))


if __name__ == "__main__":
    vertical_distance_between_layers = 6
    horizontal_distance_between_neurons = 2
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 4
    network = NeuralNetwork()
    # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)

    nodes = [13, 10, 7, 3]
    weights = []
    for i in range(len(nodes) - 1):
        weights.append(np.ones((nodes[i], nodes[i + 1])))
    print(weights)
    weights[0][0:2, :6] = -0.4
    weights[0][2:, ] = 0.5
    weights[1][0:2, :6] = -0.2
    weights[1][2:, 1:5] = -0.3
    weights[1][2:, 8:] = -0.3
    """
    weights[2][8:, 2] = -0.7
    weights[2][1:, 3:] = -0.5
    """
    for i in range(len(nodes) - 1):
        network.add_layer(nodes[i], weights[i])
    network.add_layer(nodes[-1])
    print("nodes:{}".format(nodes))
    print("weights:{}".format(np.shape(weights)))
    # print("weights:\n{}".format(weights))
    mydraw(weights, 0)

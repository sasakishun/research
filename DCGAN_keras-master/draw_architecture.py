from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np

vertical_distance_between_layers = 6
horizontal_distance_between_neurons = 2
neuron_radius = 0.5
number_of_neurons_in_widest_layer = 4
from datetime import datetime
import cv2

max_weight = 0
class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, text=""):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        _text = pyplot.text(self.x-0.25, self.y-0.25, text, fontsize=neuron_radius*10)
        pyplot.gca()._add_text(_text)
        pyplot.gca().add_patch(circle)

class Layer():
    def __init__(self, network, number_of_neurons, weights):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

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
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=abs(linewidth), color=color, alpha=abs(linewidth))
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw(this_layer_neuron_index)
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    # print("weight:{}".format(weight))
                    print("alpha:{}".format(weight / np.max(abs(self.previous_layer.weights))))
                    # print("weight:{}".format(self.previous_layer.weights))
                    # print("max_weight:{}".format(np.max(abs(self.previous_layer.weights))))
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, linewidth=weight/np.max(abs(self.previous_layer.weights)))
                    # weightはスカラー値であり、重みの大きさ＝結合の太さ(linewidth)とする
                    # 正の重みは赤、負の重みは青で表示
                    # linewidthだと整数値しか扱えない -> 透明度で結合強度を表現した方がいい


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    """
    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()
    """
    def draw(self, path, acc=-1):
        for layer in self.layers:
            layer.draw()
        pyplot.title("acc:{:.4f}".format(acc))
        pyplot.axis('scaled')
        pyplot.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False)
        pyplot.xlabel("input")
        path += "{}{}_{}".format(r"\test", r"\{}".format(datetime.now().strftime("%Y%m%d%H%M%S")), "architecture")
        pyplot.savefig(path, bbox_inches="tight", pad_inches=0.0)
        print("saved to -> {}".format(path))
        # pyplot.show()
        pyplot.close()
        return cv2.imread(path + ".png")

def mydraw(_weights, acc):
    vertical_distance_between_layers = 6
    horizontal_distance_between_neurons = 2
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 4
    network = NeuralNetwork()
    # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)

    weights = []
    nodes = []
    print("_weights:{}\n{}".format([np.shape(weight) for weight in _weights], _weights))
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

    nodes.append(np.shape(_weights[-1])[0])
    print("weights:{}\n{}".format([np.shape(weight) for weight in weights], weights))
    # for i in range(len(nodes)-1):
        # weights.append(np.ones((nodes[i+1], nodes[i])))

    print("nodes:{}".format(nodes))
    print("weights:{}".format(np.shape(weights)))

    for i in range(len(nodes)-1):
        network.add_layer(nodes[i], weights[i].T)
    network.add_layer(nodes[-1])
    # print("weights:\n{}".format(weights))
    path = r"C:\Users\papap\Documents\research\DCGAN_keras-master\visualized_iris\network_architecture"
    return network.draw(path=path, acc=acc)

if __name__ == "__main__":
    vertical_distance_between_layers = 6
    horizontal_distance_between_neurons = 2
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 4
    network = NeuralNetwork()
    # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)

    nodes = [4, 2, 3]
    weights = []
    for i in range(len(nodes)-1):
        weights.append(np.ones((nodes[i+1], nodes[i])))
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
    for i in range(len(nodes)-1):
        network.add_layer(nodes[i], weights[i])
    network.add_layer(nodes[-1])
    print("nodes:{}".format(nodes))
    print("weights:{}".format(np.shape(weights)))
    # print("weights:\n{}".format(weights))
    network.draw()

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
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
        if linewidth > 0:
            color = "red"
        else:
            color = "blue"
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=1, color=color, alpha=abs(linewidth/2))
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    print("weight:{}".format(weight))
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, linewidth=weight)
                    # weightはスカラー値であり、重みの大きさ＝結合の太さ(linewidth)とする
                    # 正の重みは赤、負の重みは青で表示
                    # linewidthだと整数値しか扱えない -> 透明度で結合強度を表現した方がいい


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()


if __name__ == "__main__":
    vertical_distance_between_layers = 6
    horizontal_distance_between_neurons = 2
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 4
    network = NeuralNetwork()
    # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)
    weights1 = np.ones((10, 4))
    """
    np.array([[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 1, 1],
             [0, 1, 0, 1],
             [0, 0, 0, 0],
             [0, 0, 1, 1],
             [0, 1, 0, 1],
             [0, 0, 0, 0],
             [0, 0, 1, 1],
             [0, 1, 0, 1]])
     """
    weights2 = np.ones((3, 10))

    weights2[0:2, :6] = -0.5
    weights2[0:1, 1] = -2
    """
    np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],])
    """
    network.add_layer(4, weights1)
    network.add_layer(10, weights2)
    network.add_layer(3)
    network.draw()

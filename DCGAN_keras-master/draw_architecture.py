from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np
import matplotlib.patches as pat

vertical_distance_between_layers = 60  # 6
horizontal_distance_between_neurons = 10  # 2
neuron_radius = 0.5
number_of_neurons_in_widest_layer = 4
from datetime import datetime
import cv2
import os

max_weight = 0
colors = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
colors = [colors[0],
          colors[8],
          colors[3],
          colors[1],
          colors[2],
          colors[4],
          colors[5],
          colors[6],
          colors[7],
          colors[9]]  # 色指定


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, text="", color=None, annotation=None, label_class=None):
        global vertical_distance_between_layers
        global horizontal_distance_between_neurons
        _neuron_radius = neuron_radius * 30
        # 分類クラス数が多い場合は、分類先以外のクラスをグレー塗り
        if annotation is not None:
            if len(annotation) == 10 and label_class is not None:
                print("label_class:{}".format(label_class))
                print("annotation:{}".format(annotation))
                # annotation = [annotation[label_class], np.sum(annotation[:label_class]) + np.sum(annotation[label_class+1:])]
                _colors = [colors[i] if i == label_class else "white" for i in range(len(annotation))]
            else:
                _colors = colors[:len(annotation)]
            if np.sum(annotation) == 0:
                annotation = [1 / len(annotation) for _ in annotation]
            pyplot.pie(x=annotation,
                       radius=_neuron_radius,
                       counterclock=False,
                       center=(self.x, self.y),
                       colors=_colors,
                       startangle=90,
                       # labels=["{:.2f}%".format(_annotation * 100) for _class, _annotation in enumerate(annotation)],
                       # textprops = {'fontsize': neuron_radius*5}
                       )
            # pyplot.gca().add_patch(pyplot.Circle((self.x, self.y), radius=_neuron_radius, fill=True, color="gray"))
            pyplot.gca().add_patch(pyplot.Circle((self.x, self.y), radius=_neuron_radius, fill=False, color="black"))
            pyplot.gca().add_patch(pyplot.Circle((self.x, self.y), radius=0.1, fill=True, color="black"))
            # pyplot.gca().add_patch(p)

            if False:
                for _class, _annotation in enumerate(annotation):
                    if _annotation is not None:
                        _text_annotation = pyplot.text(self.x,
                                                       self.y - 2 - neuron_radius * 10 * _class,
                                                       # vertical_distance_between_layers * _class / len(annotation),
                                                       "{}: {:.2f}%".format(chr(_class + ord("A")),
                                                                            _annotation * 100),
                                                       fontsize=neuron_radius * 5, color="black")
                        pyplot.gca()._add_text(_text_annotation)
                    if False:
                        # 中心座標(0.5, 0.5), 半径0.4, 切込み位置0°, 60°
                        if _annotation > 0.01:
                            w = pat.Wedge(center=(self.x, self.y), r=_neuron_radius,
                                          theta1=90 - 360 * sum(annotation[:_class + 1]),
                                          theta2=90 - 360 * sum(annotation[:_class]),
                                          color=colors[_class],
                                          edgecolor="white")
                            # Axesに扇形を追加
                            pyplot.gca().add_patch(w)

                    # 円を表示する場合
                    if False:
                        circle = pyplot.Circle((self.x, self.y),
                                               radius=_neuron_radius * sum(annotation[:_class + 1]),  # 5,
                                               facecolor=colors[_class], edgecolor=colors[_class])
                        pyplot.gca().add_patch(circle)
        # 異常ノードの場合(白は数値のみ描画するために使用)
        else:
            if color is None:
                color = [{"color": "black"}]
            for i, _color in enumerate(color):
                """
                _text_correct_range = pyplot.text(self.x + 4, self.y - 12, "[{:.2f}, {:.2f}]".format(
                    _color["correct_range"][0], _color["correct_range"][1]) if "correct_range" in _color else None,
                                                  fontsize=neuron_radius * 10, color="gray")
                pyplot.gca()._add_text(_text_correct_range)
                """
                if _color["color"] != "black":  # ミスニューロンは半径大きく、黒以外で描画
                    slip = 3 * (i - 1)
                    if _color["color"] != "white":
                        circle = pyplot.Circle((self.x + slip, self.y + slip), radius=neuron_radius * 5,
                                               facecolor=_color["color"], edgecolor=_color["color"],
                                               width=0.5)
                        # 異常ノードの場合(白は数値のみ描画するために使用)
                        _text_correct_range = pyplot.text(self.x + 4, self.y - 12, "[{:.2f}, {:.2f}]".format(
                            _color["correct_range"][0],
                            _color["correct_range"][1]) if "correct_range" in _color else None,
                                                          fontsize=neuron_radius * 10, color="gray")
                        pyplot.gca()._add_text(_text_correct_range)

                    _text_value = pyplot.text(self.x + 4, self.y - 4,
                                              "{:.2f}".format(_color["value"]) if "value" in _color else None,
                                              fontsize=neuron_radius * 10, color="gray")
                    pyplot.gca()._add_text(_text_value)
                else:
                    circle = pyplot.Circle((self.x, self.y), radius=_neuron_radius, fill=False, color="black")
                # _text = pyplot.text(self.x-0.25, self.y-0.25, text, fontsize=neuron_radius*10)
                # pyplot.gca()._add_text(_text)
                pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, weights, non_active_neurons=None, node_color=None, annotation=None,
                 label_class=None, is_image=False):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons, is_image)
        self.weights = weights
        self.neuron_color = node_color if node_color is not None else [[{"color": "black"}] for _ in
                                                                       range(number_of_neurons)]
        self.annotation = annotation
        self.label_class = label_class
        if non_active_neurons:
            for i in non_active_neurons:
                non_active_neurons[i] = "white"

    # 可視化で重要
    def __intialise_neurons(self, number_of_neurons, is_image=False):
        # image_flag == Trueだと入力:64を8*8にして可視化
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(
            int(np.sqrt(number_of_neurons) if is_image else number_of_neurons))
        for iteration in range(number_of_neurons):
            if is_image:
                if iteration % int(np.sqrt(number_of_neurons)) == 0:
                    x = self.__calculate_left_margin_so_layer_is_centered(np.sqrt(number_of_neurons))
                neuron = Neuron(x, self.y -
                                horizontal_distance_between_neurons * np.sqrt(number_of_neurons)
                                * (iteration // np.sqrt(number_of_neurons))
                                - horizontal_distance_between_neurons)  # 40)
                neurons.append(neuron)
                x += horizontal_distance_between_neurons * (64 / np.sqrt(number_of_neurons))
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

        color = "black"

        linewidth = abs(linewidth)
        if linewidth > 0.:
            linewidth = max(0.4, linewidth)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=abs(linewidth), color=color, alpha=abs(linewidth))
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            # neuron.draw(text=this_layer_neuron_index, color=self.neuron_color[this_layer_neuron_index])
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    # weightsを転置しているため、weights[接続先親ノード番号、子ノード番号]が重みとなる
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    # 親が同じ重み同士で、表示重みの太さを割合化
                    _sum = np.sum([abs(i) ** 4 for i in self.previous_layer.weights[this_layer_neuron_index]])
                    # print("weight[?][{}] -> sum:{}".format(this_layer_neuron_index, _sum))
                    # print(self.previous_layer.weights[this_layer_neuron_index])
                    linewidth = 2 * np.sign(weight) * weight ** 4 / _sum if _sum > 0 else 0
                    # linewidth = weight / np.max(abs(self.previous_layer.weights))
                    # print("weight:{}".format(weight))
                    # print("alpha:{}".format(weight / np.max(abs(self.previous_layer.weights))))
                    # print("weight:{}".format(self.previous_layer.weights))
                    # print("max_weight:{}".format(np.max(abs(self.previous_layer.weights))))
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, linewidth=linewidth)
                    # weightはスカラー値であり、重みの大きさ＝結合の太さ(linewidth)とする
                    # 正の重みは赤、負の重みは青で表示
                    # linewidthだと整数値しか扱えない -> 透明度で結合強度を表現した方がいい
            neuron.draw(text=this_layer_neuron_index, color=self.neuron_color[this_layer_neuron_index],
                        annotation=self.annotation[this_layer_neuron_index] if self.annotation is not None else None,
                        label_class=self.label_class)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None, non_active_neurons=None, node_color=None, annotation=None,
                  label_class=None, is_image=False):
        layer = Layer(self, number_of_neurons, weights, non_active_neurons, node_color=node_color,
                      annotation=annotation, label_class=label_class,
                      is_image=is_image)
        self.layers.append(layer)

    """
    def draw(self):
        for layer in self.layers:
            layer.draw()
        pyplot.axis('scaled')
        pyplot.show()
    """

    def draw(self, path, acc=None, comment="", dir=None):
        for layer in self.layers:
            layer.draw()
        if acc is not None:
            comment += " acc:{:.4f}".format(acc)
        # pyplot.title(comment)
        pyplot.axis('scaled')
        pyplot.tick_params(labelbottom=False,
                           labelleft=False,
                           labelright=False,
                           labeltop=False)
        # pyplot.xlabel("input")
        if not os.path.exists(path):  # ディレクトリがないとき新規作成
            path = os.getcwd() + r"\visualized_iris\network_architecture"
            print("path:{}".format(path))
        if dir is not None:
            from visualization import my_makedirs
            path = os.getcwd() + r"\visualized_iris\hidden_output" + dir
            my_makedirs(path)
            path += r"\{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        else:
            path += "{}{}_{}".format(r"\test", r"\{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
                                     "architecture")
        for _frame in ["right", "top", "bottom", "left"]:
            pyplot.gca().spines[_frame].set_visible(False)
        pyplot.tick_params(color='white')
        pyplot.savefig(path, bbox_inches="tight", pad_inches=0.0, dpi=2000)
        print("saved to -> {}".format(path))
        # pyplot.show()
        pyplot.close()
        return cv2.imread(path + ".png")


# 中間層出力が0に繋がる重みを削除
# ～（応用例）～
# このintermidate_outputsをmaskと考え、消したいノードだけを0、他を1にすれば
# いらないノード重みとその子孫だけ消せる
def delet_kernel_with_intermidate_out_is_zero(_kernel, intermidate_outpus):
    _kernel = [i for i in _kernel if i.ndim == 2]
    for parent_layer in reversed(range(1, len(intermidate_outpus) - 1)):
        for parent_node in range(len(intermidate_outpus[parent_layer])):
            if intermidate_outpus[parent_layer][parent_node] == 0:
                # 出力0に繋がる重みを0初期化、子ノード出力も0に
                for _child in range(len(intermidate_outpus[parent_layer - 1])):
                    if _kernel[parent_layer - 1][_child][parent_node] != 0:
                        print("parent_layer:{} parent_nodes:{} _child:{} parent_out:{}"
                              .format(parent_layer, parent_node, _child, intermidate_outpus[parent_layer][parent_node]))
                        # 親に繋がる子ノード出力を0に(入力層以外)
                        if parent_layer - 1 > 0:
                            intermidate_outpus[parent_layer - 1][_child] = 0
                        # 子から親に繋がる重みも0に
                        _kernel[parent_layer - 1][_child][parent_node] = 0
    # 入力層から順に、子ノードとの結合がないノードの重み削除
    for parent_layer in range(1, len(intermidate_outpus) - 1):
        for parent_node in range(len(intermidate_outpus[parent_layer])):
            # 子ノードと結合があるか確認
            connect = False
            for child_node in range(len(intermidate_outpus[parent_layer - 1])):
                if _kernel[parent_layer - 1][child_node][parent_node] != 0:
                    connect = True
                    break
            if not connect:
                for parent_parent_node in range(len(intermidate_outpus[parent_layer + 1])):
                    _kernel[parent_layer][parent_node][parent_parent_node] = 0
    return _kernel


# 子ノードと結合がある重みのみ抽出
def extruct_weight_of_target_class(_weights, target_class, annotation=None):
    weights = []
    for i in range(np.shape(_weights)[0]):
        if _weights[i].ndim == 2:
            weights.append(_weights[i])
    _weights = weights

    # target_class以外の重みを削除
    parents = [target_class]
    weights = [np.zeros(np.shape(i)) for i in _weights]
    for layer, _weight in reversed(list(enumerate(_weights))):
        child = []
        for i in range(len(_weight)):
            for j in parents:
                if _weight[i][j] != 0:
                    weights[layer][i][j] = _weight[i][j]
                    child.append(i)
            if annotation is not None:
                for _layer, _annotation in enumerate(annotation):
                    print("annotation[{}]:{}".format(_layer, _annotation))
                # parents以外の親ノード
                for j in (set(list(range(np.shape(_weight)[1]))) - set(parents)):
                    print("layer:{} j:{}".format(layer, j))
                    annotation[layer + 1][j] = [None for _ in annotation[layer + 1][j]]
        parents = child
    # 入力層は消さないでおく
    if False:
        for i in (set(list(range(np.shape(weights[0])[0]))) - set(parents)):
            annotation[0][i] = [None for _ in annotation[0][i]]

    # weightsとannotationのNoneの部分を削除
    for layer in range(len(weights)):
        node = 0
        while node < len(weights[layer]):
            if annotation[layer][node][0] is None:
                print("annotation[{}][{}] is deleted".format(layer, node))
                del annotation[layer][node]
                for _weight_ in weights[layer]:
                    print(_weight_)
                weights[layer] = np.delete(weights[layer], node, 0)
                if layer > 0:
                    weights[layer - 1] = np.delete(weights[layer - 1], node, 1)
                print("deleted")
            else:
                node += 1

    # 出力層ノードも削除
    del annotation[-1][:target_class]
    del annotation[-1][1:]
    for i in range(target_class):
        weights[-1] = np.delete(weights[-1], 0, 1)
    for i in weights:
        print(i)
    for i in range(1, np.shape(weights[-1])[1]):
        weights[-1] = np.delete(weights[-1], 1, 1)
    print("deleted_weights")
    for i in weights:
        print(i)
    print("deleted_annotation")
    for i in annotation:
        print(i)

    return weights, annotation


def mydraw(_weights, acc=None, comment="", non_active_neurons=None, node_colors=None, dir=None, annotation=None,
           target_class=None, label_class=None):
    neuron_radius = 0.5
    number_of_neurons_in_widest_layer = 4
    pyplot.tick_params(labelbottom=False,
                       labelleft=False,
                       labelright=False,
                       labeltop=False)

    print("_weights:{}".format(_weights))
    _weights = [i for i in _weights if i.ndim == 2]  # kernelだけ抽出
    # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)
    if target_class is not None:
        _weights, annotation = extruct_weight_of_target_class(_weights, target_class, annotation=annotation)
        for i in _weights:
            if len(i) == 0:
                return
    weights = []
    nodes = []
    for i in range(np.shape(_weights)[0]):
        if _weights[i].ndim == 2:
            nodes.append(np.shape(_weights[i])[0])
            weights.append(_weights[i])
    nodes.append(np.shape(weights[-1])[1])
    print("nodes of each layer:{}".format(nodes))

    global vertical_distance_between_layers
    vertical_distance_between_layers = nodes[-1] * 100
    global horizontal_distance_between_neurons
    if nodes[0] == 64 or nodes[0] == 784:
        horizontal_distance_between_neurons = np.sqrt(nodes[0]) / 2
    else:
        horizontal_distance_between_neurons = nodes[0] / 2  # + 2 # ノード間隔をあけるためのマージン
    print("vertical_distance_between_layers:{}".format(vertical_distance_between_layers))
    print("horizontal_distance_between_neurons:{}".format(horizontal_distance_between_neurons))

    network = NeuralNetwork()
    for i in range(len(nodes) - 1):
        network.add_layer(nodes[i], weights[i].T, non_active_neurons[i] if non_active_neurons is not None else None,
                          node_color=node_colors[i] if node_colors is not None else None,
                          annotation=annotation[i] if annotation is not None else None,
                          label_class=label_class,
                          is_image=True if (i == 0 and nodes[0] == 64) else False)
    network.add_layer(nodes[-1], node_color=node_colors[-1] if node_colors is not None else None,
                      annotation=annotation[-1] if annotation is not None else None,
                      label_class=label_class)
    path = os.getcwd() + r"\visualized_iris\network_architecture"
    return network.draw(path=path, acc=acc, comment=comment, dir=dir)


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
    # vertical_distance_between_layers = 6
    # horizontal_distance_between_neurons = 2
    # neuron_radius = 0.5
    # number_of_neurons_in_widest_layer = 4
    network = NeuralNetwork()
    # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)

    nodes = [8, 4, 2, 1]
    weights = []
    for i in range(len(nodes) - 1):
        weights.append(np.zeros((nodes[i], nodes[i + 1])))
    for _weight in weights:
        for i in range(np.shape(_weight)[0]):
            # for j in range(np.shape(_weight)[1]):
            _weight[i][i // 2] = 1
    print(weights)
    weights[0][0, 0] = 2
    weights[0][4, 2] = 2
    weights[0][5, 2] = 2
    weights[0][7, 3] = 2
    weights[1][0, 0] = 4
    weights[1][2, 1] = 4
    # weights[1][3, 1] = 4
    weights[2][0, 0] = 4
    annotation = [[[0, 0, 0] for _ in range(_node)] for _node in nodes]
    annotation[0][0] = [1, 0.8, 0.9]
    annotation[0][1] = [0.8, 0.9, 1]
    annotation[0][2] = [1.2, 1, 1]
    annotation[0][3] = [1.2, 1, 1]
    annotation[0][4] = [1.3, 1, 1]
    annotation[0][5] = [1.3, 0.9, 1.2]
    annotation[0][6] = [0.5, 1, 0.8]
    annotation[0][7] = [1.3, 0.7, 0.9]
    for layer in range(len(weights)):
        for node in range(np.shape(weights[layer])[1]):
            print("node:{} child:{} {}".format(node, node * 2, node * 2 + 1))
            for i in range(3):
                annotation[layer + 1][node][i] = \
                    annotation[layer][node * 2][i] * weights[layer][node * 2][node] + \
                    annotation[layer][node * 2 + 1][i] * weights[layer][node * 2 + 1][node]
            for i in range(3):
                if annotation[layer + 1][node][i] == max(annotation[layer + 1][node]):
                    annotation[layer + 1][node][i] *= 2
                if annotation[layer + 1][node][i] == min(annotation[layer + 1][node]):
                    annotation[layer + 1][node][i] /= 2
                if annotation[layer + 1][node][i] < 0.1:
                    annotation[layer + 1][node][i] = 0

    for i in annotation:
        print(i)
    mydraw(weights, annotation=annotation, label_class=0)
    exit()
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

from _model_weightGAN import *
import cv2
from time import *
import datetime
from draw_architecture import *
from binary__tree_main import divide_data
import copy
from visualization import *

def stringToList(_string, split=" "):
    str_list = []
    temp = ''
    for x in _string:
        if x == split:  # 区切り文字
            str_list.append(temp)
            temp = ''
        else:
            temp += x
    if temp != '':  # 最後に残った文字列を末尾要素としてリストに追加
        str_list.append(temp)
    return str_list

def convert_weigths_tree2mlp(tree_weights, mlp_weights, input_size, output_size, dense_size):
    ### tree構造の重みを全結合モデルにセット
    tree_shape = tree(input_size, output_size, get_hidden_flag=True)
    print("tree_shape:{}".format(tree_shape))
    print("input_size:{} output_size:{}".format(input_size, output_size))  # 13, 3
    tree_index = 0
    for i in range(output_size):
        for j in range(len(tree_shape)):
            for k in range(tree_shape[j]):
                _node = k + i * (dense_size + [output_size])[j + 1] // output_size  # tree_shape[j]
                # print("i:{} j:{} k:{} _node:{}".format(i, j, k, _node))
                for l in range(2):
                    mlp_weights[2 * j][_node * 2 + l][_node] \
                        = tree_weights[tree_index][l][0]
                tree_index += 1
                mlp_weights[2 * j + 1][_node] = tree_weights[tree_index]
                tree_index += 1
    ### tree構造の重みを全結合モデルにセット

    ### このままだと途中の中間層へ直接入力されるノードには入力が伝播しない->これを解決
    for i in range(output_size):
        last_node = input_size + i * (mlp_weights[0].shape[0] // output_size) - 1
        for layer in range(len(mlp_weights)):
            if mlp_weights[layer].ndim == 2:
                if last_node % 2 == 0:
                    mlp_weights[layer][last_node][last_node // 2] = 1.
                    print("input fixing weight[{}][{}][{}] == 1".format(layer, last_node, last_node // 2))
                    last_node = last_node // 2
    ### このままだと途中の中間層へ直接入力されるノードには入力が伝播しない->これを解決
    return mlp_weights


def add_original_input(input_size, output_size, mlp_weights):
    ### tree->mlpにする際に複製したノードに繋がる基入力層を追加する
    input_weight = np.zeros((input_size, mlp_weights[0].shape[0]))
    input_bias = np.zeros((mlp_weights[0].shape[0],))
    for i in range(output_size):
        for j in range(input_size):
            input_weight[j][j + i * (mlp_weights[0].shape[0] // output_size)] = 1
    ### tree->mlpにする際に複製したノードに繋がる基入力層を追加する
    return [input_weight] + [input_bias] + mlp_weights


def non_active_in_tree2mlp(input_size, output_size, dense_size):
    # 入力 : mlp_weights
    # 出力 : 各mlp層の不使用ノードのリスト(non_active_neurons)
    # 処理 : 「treeのノード数、mlpのノード数」の差分mlpノード番号をnon_active_neuronsとして格納
    tree_shape = tree(input_size, output_size, get_hidden_flag=True)
    non_active_neurons = []
    for i in range(output_size):
        for j in range(len(tree_shape)):
            for k in range(tree_shape[j]):
                _node = k + i * (dense_size + [output_size])[j + 1] // output_size

    return non_active_neurons


def mask(masks, batch_size):
    ###各maskをミニバッチサイズでそれぞれ複製
    return [np.array([masks[i] for _ in range(batch_size)]) for i in range(len(masks))]
    # mask[0]*32, mask[1]*32,...を返す


def inputs_z(X_test, g_mask_1):
    return list(np.array([X_test])) + mask(g_mask_1, len(X_test))


# 入力: masked_mlpモデル 出力:不要ノードを削除したmasked_mlpモデル
def shrink_tree_nodes(model, target_layer, X_train, y_train, X_test, y_test, only_active_list=False):
    # model: maskd_mlp
    # 入力 : 全クラス分類モデル(model)、対象レイヤー番号(int)、訓練データ(np.array)、訓練ラベル(np.array)
    # 出力 : 不要ノードを削除したモデル(model)
    target_layer *= 2  # weigthsリストが[重み、バイアス....]となっているため
    weights = model.get_weights()
    dense_size = [weights[0].shape[0]] + [weights[i * 2 + 1].shape[0] for i in
                                          range(len(weights) // 2)]  # [13, 48, 24, 12, 6, 3]
    output_size = weights[-1].shape[0]
    _mask = [np.array([1 for _ in range(dense_size[i])]) for i in range(len(dense_size))]
    active_nodes = [[] for _ in range(output_size)]
    X_trains, y_trains = divide_data(X_train, y_train, dataset_category=output_size)  # クラス別に訓練データを分割
    for i in range(output_size):  # for i in range(クラス数):
        print("inputs_z(X_trains[{}], _mask):{}".format(i, [np.shape(j) for j in inputs_z(X_trains[i], _mask)]))
        # i クラスで使用するactiveノード検出 -> active_nodes=[[] for _ in range(len(クラス数))]
        pruned_train_val_acc = model.evaluate(inputs_z(X_trains[i], _mask), y_trains[i])[1]
        for j in range(len(_mask[target_layer // 2])):
            _mask[target_layer // 2][j] = 0
            _acc = model.evaluate(inputs_z(X_trains[i], _mask), y_trains[i])[1]
            if _acc < pruned_train_val_acc * 0.999:
                active_nodes[i].append(j)  # activeノードの番号を保存 -> active_nodes[i].append(activeノード)
            _mask[target_layer // 2][j] = 1
    print("active_nodes:{}".format(active_nodes))
    if only_active_list:
        return active_nodes
    usable = [True for _ in range(len(_mask[target_layer // 2]))]  # ソートに使用済みのノード番号リスト
    altered_weights = [[], [], []]  # [np.zeros((weights[target_layer]).shape),
    for i in range(output_size):
        for j in range(len(active_nodes[i])):
            if usable[active_nodes[i][j]]:
                altered_weights[0].append(weights[target_layer - 2][:, active_nodes[i][j]])
                altered_weights[1].append(weights[target_layer - 1][active_nodes[i][j]])
                altered_weights[2].append(weights[target_layer][active_nodes[i][j]])
            usable[active_nodes[i][j]] = False
    for i in range(len(altered_weights)):
        altered_weights[i] = np.array(altered_weights[i])
        if i == 0:
            altered_weights[0] = altered_weights[0].T
    weights[target_layer - 2] = altered_weights[0][:, :sum(1 for x in usable if not x)]
    weights[target_layer - 1] = altered_weights[1][:sum(1 for x in usable if not x)]
    weights[target_layer] = altered_weights[2][:sum(1 for x in usable if not x)]
    # ここで第[target_layer // 2]中間層のノード数を変更し
    # モデルを再定義
    dense_size[target_layer // 2] = sum(1 for x in usable if not x)
    _mlp = masked_mlp(dense_size[0], dense_size[1:-1], output_size)
    _mlp.set_weights(weights)

    _mask = [np.array([1 for _ in range(dense_size[i])]) for i in range(len(dense_size))]
    intermediate_layer_model = [Model(inputs=_mlp.input,
                                     outputs=_mlp.get_layer("dense{}".format(i)).output) for i in range(len(dense_size)-1)]
    intermediate_output = [intermediate_layer_model[i].predict(inputs_z([X_train[0]], _mask)) for i in range(len(dense_size)-1)]
    for i in range(len(intermediate_output)):
        print("dense[{}]:{}".format(i, intermediate_output[i]))
    visualize_network(weights, _mlp.evaluate(inputs_z(X_test, _mask), y_test)[1],
                      comment="shrinking layer[{}]".format(target_layer // 2))
    return _mlp

def shrink_mlp_nodes(model, target_layer, X_train, y_train, X_test, y_test, only_active_list=False):
    # model: _mlp
    # 入力 : 全クラス分類モデル(model)、対象レイヤー番号(int)、訓練データ(np.array)、訓練ラベル(np.array)
    # 出力 : 不要ノードを削除したモデル(model)
    target_layer *= 2  # weigthsリストが[重み、バイアス....]となっているため
    weights = model.get_weights()
    dense_size = [weights[0].shape[0]] + [weights[i * 2 + 1].shape[0] for i in
                                          range(len(weights) // 2)]  # [13, 48, 24, 12, 6, 3]
    output_size = weights[-1].shape[0]
    _mask = [np.array([1 for _ in range(dense_size[i])]) for i in range(len(dense_size))]
    active_nodes = [[] for _ in range(output_size)]
    X_trains, y_trains = divide_data(X_train, y_train, dataset_category=output_size)  # クラス別に訓練データを分割
    for i in range(output_size):  # for i in range(クラス数):
        print("inputs_z(X_trains[{}], _mask):{}".format(i, [np.shape(j) for j in inputs_z(X_trains[i], _mask)]))
        # i クラスで使用するactiveノード検出 -> active_nodes=[[] for _ in range(len(クラス数))]
        pruned_train_val_acc = model.evaluate(inputs_z(X_trains[i], _mask), y_trains[i])[1]
        for j in range(len(_mask[target_layer // 2])):
            _mask[target_layer // 2][j] = 0
            _acc = model.evaluate(inputs_z(X_trains[i], _mask), y_trains[i])[1]
            if _acc < pruned_train_val_acc * 0.999:
                active_nodes[i].append(j)  # activeノードの番号を保存 -> active_nodes[i].append(activeノード)
            _mask[target_layer // 2][j] = 1
    print("active_nodes:{}".format(active_nodes))
    if only_active_list:
        return active_nodes
    usable = [True for _ in range(len(_mask[target_layer // 2]))]  # ソートに使用済みのノード番号リスト
    altered_weights = [[], [], []]  # [np.zeros((weights[target_layer]).shape),

    for i in range(1, len(active_nodes)):
        active_nodes[0] += active_nodes[i]
        active_nodes[i] = []
    active_nodes[0] = sorted(set(active_nodes[0]))
    if len(active_nodes[0]) == 0:
        return model
    for i in range(output_size):
        for j in range(len(active_nodes[i])):
            if usable[active_nodes[i][j]]:
                altered_weights[0].append(weights[target_layer - 2][:, active_nodes[i][j]])
                altered_weights[1].append(weights[target_layer - 1][active_nodes[i][j]])
                altered_weights[2].append(weights[target_layer][active_nodes[i][j]])
            usable[active_nodes[i][j]] = False
    for i in range(len(altered_weights)):
        altered_weights[i] = np.array(altered_weights[i])
        if i == 0:
            altered_weights[0] = altered_weights[0].T
    weights[target_layer - 2] = altered_weights[0][:, :sum(1 for x in usable if not x)]
    weights[target_layer - 1] = altered_weights[1][:sum(1 for x in usable if not x)]
    weights[target_layer] = altered_weights[2][:sum(1 for x in usable if not x)]
    # ここで第[target_layer // 2]中間層のノード数を変更し
    # モデルを再定義
    dense_size[target_layer // 2] = sum(1 for x in usable if not x)
    _mlp = masked_mlp(dense_size[0], dense_size[1:-1], output_size)
    _mlp.set_weights(weights)

    _mask = [np.array([1 for _ in range(dense_size[i])]) for i in range(len(dense_size))]
    intermediate_layer_model = [Model(inputs=_mlp.input,
                                     outputs=_mlp.get_layer("dense{}".format(i)).output) for i in range(len(dense_size)-1)]
    intermediate_output = [intermediate_layer_model[i].predict(inputs_z([X_train[0]], _mask)) for i in range(len(dense_size)-1)]
    for i in range(len(intermediate_output)):
        print("dense[{}]:{}".format(i, intermediate_output[i]))
    visualize_network(weights, _mlp.evaluate(inputs_z(X_test, _mask), y_test)[1],
                      comment="shrinking layer[{}]".format(target_layer // 2))
    return _mlp

def visualize_network(weights, acc=-1, comment="", non_active_neurons=None):
    print("visualising_start")
    # return
    from time import sleep
    sleep(1)
    # return
    im_architecture = mydraw(weights, acc, comment, non_active_neurons)
    im_h_resize = im_architecture
    path = os.getcwd() + r"\visualized_iris\network_architecture\triple\{}".format(
        datetime.now().strftime("%Y%m%d%H%M%S") + ".png")
    cv2.imwrite(path, im_h_resize)
    print("saved concated graph to -> {}".format(path))
    return

def sort_weights(_weights, target_layer=None):
    weights = [_weights[2*i] for i in range(len(_weights)//2)]
    bias = [_weights[2*i+1] for i in range(len(_weights)//2)]
    sorted_weights = [np.zeros(_weights[2 * i].shape) for i in range(len(_weights) // 2)]
    sorted_bias = [np.zeros(_weights[2*i+1].shape) for i in range(len(_weights)//2)]

    # 入力層の入れ替え->独立した二分木構造を見せるため
    if target_layer == -1:
        return
    # weights[i][j][k] : i層j番ノードからi+1層k番ノードへの重み結合
    for i in range(len(weights)-1):
        if target_layer is not None:
            i = target_layer
        first_connect = [[float("inf"), None] for k in range(weights[i].shape[1])]
        for j in range(weights[i].shape[0]):
            for k in range(weights[i].shape[1]):
                if weights[i][j][k] != 0.:
                    if first_connect[k][1] is None:
                        first_connect[k] = [j, k]
        first_connect.sort()
        print("\ni:{} first_connect:{}".format(i, first_connect))
        ### first_connectを基にsorted_weightsを作成
        print("weights:{}".format([np.shape(i) for i in weights]))
        print("seorted_weights:{}".format([np.shape(i) for i in sorted_weights]))
        for k in range(weights[i].shape[1]):
            if first_connect[k][1] is None:
                break
            sorted_weights[i].T[k] = weights[i].T[first_connect[k][1]]
            sorted_bias[i][k] = bias[i][first_connect[k][1]]
            if i + 1 < len(sorted_weights):
                sorted_weights[i+1][k] = weights[i+1][first_connect[k][1]]
        ### weightsをソート済みweightで置換
        weights[i] = copy.deepcopy(sorted_weights[i])
        bias[i] = copy.deepcopy(sorted_bias[i])
        if i + 1 < len(weights):
            weights[i+1] = copy.deepcopy(sorted_weights[i+1])
        _weights = [weights[i // 2] if i % 2 == 0 else bias[i // 2] for i in range(len(_weights))]
        if target_layer >= 0:
            return _weights
    visualize_network(_weights, acc=-1, comment="sort all layer", non_active_neurons=None)
    return _weights


from binary__tree_main import get_kernel_and_bias
def show_intermidate_output(data, target, name, _mlp):
    np.set_printoptions(precision=3)
    # for i in range(len(get_kernel_and_bias(_mlp))):
        # print("_mlp[{}]:{}".format(i, np.shape(get_kernel_and_bias(_mlp)[i])))
    intermediate_layer_model = [Model(inputs=_mlp.input, outputs=_mlp.get_layer("dense{}".format(i)).output)
                                for i in range(len(get_kernel_and_bias(_mlp)) // 2)]
    # for i in range(len(intermediate_layer_model[-1].get_weights())):
        # print("intermidate[{}]:{}".format(i, np.shape(intermediate_layer_model[-1].get_weights()[i])))
    output_size = np.shape(get_kernel_and_bias(intermediate_layer_model[-1])[-1])[0]
    dataset_category = output_size

    original_data = copy.deepcopy(data)
    original_target = copy.deepcopy(target)
    data, target = divide_data(data, target, dataset_category)
    print("dataset_category:{}".format(dataset_category))
    for i in range(len(data)):
        print("data[{}]:{}".format(i, np.shape(data[i])))
    correct_data = [[] for _ in range(output_size)]
    correct_target = [[] for _ in range(output_size)]
    incorrect_data = [[] for _ in range(output_size)]
    incorrect_target = [[] for _ in range(output_size)]
    output = [intermediate_layer_model[-1].predict(data[i]) for i in range(output_size)]
    # for i in range(output_size):
        # print("\n\noutput[{}]:{}".format(i, np.shape(output[i])))
        # for j in range(len(output[i])):
            # print(output[i][j])
    # [masked_mlp_model.predict(inputs_z(data[i], _mask)) for i in range(output_size)]
    ### 正解データと不正解データに分割
    for i in range(output_size):
        for _data, _target, _output in zip(data[i], target[i], output[i]):
            if np.argmax(_output) == np.argmax(_target):
                correct_data[i].append(_data)
                correct_target[i].append(_target)
            else:
                incorrect_data[i].append(_data)
                incorrect_target[i].append(_target)
    for i in range(output_size):
        print("correct_data[{}]:{}".format(i, np.shape(correct_data[i])))
    for i in range(output_size):
        print("incorrect_data[{}]:{}".format(i, np.shape(incorrect_data[i])))

    ### 正解データと不正解データに分割
    correct_intermediate_output = [[list(intermediate_layer_model[i].predict([correct_data[j]]))
                                    if len(correct_data[j]) > 0 else []
                                    for j in range(len(correct_data))]
                                   for i in range(len(get_kernel_and_bias(_mlp)) // 2)]
    incorrect_intermediate_output = [[list(intermediate_layer_model[i].predict([incorrect_data[j]]))
                                      if len(incorrect_data[j]) > 0 else []
                                      for j in range(len(incorrect_data))]
                                     for i in range(len(get_kernel_and_bias(_mlp)) // 2)]

    ###入力を可視化
    labels = ["class:{}".format(i) for i in range(output_size)] \
             + ["missed_class:{}".format(i) for i in range(output_size)]
    visualize(correct_data + incorrect_data,
              None, labels, ite=cf.Iteration,
              testflag=True if name == "test" else False, showflag=False,
              comment="layer:{} {}_acc:{:.4f}".format(0, name, _mlp.evaluate(original_data, original_target)[1]))    ###入力を可視化

    ###中間層出力を可視化
    import time
    for i in range(len(get_kernel_and_bias(_mlp)) // 2):
        time.sleep(1)
        visualize(correct_intermediate_output[i] + incorrect_intermediate_output[i],
                  None, labels, ite=cf.Iteration,
                  testflag=True if name == "test" else False, showflag=False,
                  comment="layer:{} {}_acc:{:.2f}".format(i + 1, name, _mlp.evaluate(original_data, original_target)[1]))
    ###中間層出力を可視化
    for i in range(dataset_category):
        print("acc class[{}]:{}".format(i, _mlp.evaluate(data[i], target[i])[1]))
    print("toral acc: {}\n\n".format(_mlp.evaluate(original_data, original_target)[1]))
    ### 各層出力を可視化

    return concate_elements(correct_data), concate_elements(correct_target), \
           concate_elements(incorrect_data), concate_elements(incorrect_target)

def show_intermidate_train_and_test(train_data, train_target, test_data, test_target, _mlp, name=["train", "test"]):
    np.set_printoptions(precision=3)
    intermediate_layer_model = [Model(inputs=_mlp.input, outputs=_mlp.get_layer("dense{}".format(i)).output)
                                for i in range(len(get_kernel_and_bias(_mlp)) // 2)]
    output_size = np.shape(get_kernel_and_bias(intermediate_layer_model[-1])[-1])[0]
    dataset_category = output_size

    train_data, train_target = divide_data(train_data, train_target, dataset_category)
    test_data, test_target = divide_data(test_data, test_target, dataset_category)
    train_data = [list(i) for i in list(train_data)]
    test_data = [list(i) for i in list(test_data)]
    # print("dataset_category:{}".format(dataset_category))


    ### 正解データと不正解データに分割
    train_intermediate_output = [[list(intermediate_layer_model[i].predict([train_data[j]]))
                                    if len(train_data[j]) > 0 else []
                                    for j in range(len(train_data))]
                                   for i in range(len(get_kernel_and_bias(_mlp)) // 2)]
    test_intermediate_output = [[list(intermediate_layer_model[i].predict([test_data[j]]))
                                      if len(test_data[j]) > 0 else []
                                      for j in range(len(test_data))]
                                     for i in range(len(get_kernel_and_bias(_mlp)) // 2)]

    ###入力を可視化
    labels = [name[0]+"_class:{}".format(i) for i in range(output_size)] \
             + [name[1]+"_class:{}".format(i) for i in range(output_size)]
    visualize(train_data + test_data,
              None, labels, ite=cf.Iteration,
              testflag=True, showflag=False,
              comment="layer:{} input".format(0))
    ###入力を可視化

    ###中間層出力を可視化
    import time
    for i in range(len(get_kernel_and_bias(_mlp)) // 2):
        time.sleep(1)
        visualize(train_intermediate_output[i] + test_intermediate_output[i],
                  None, labels, ite=cf.Iteration,
                  testflag=True, showflag=False,
                  comment="layer:{}".format(i + 1))
    ###中間層出力を可視化
    # for i in range(dataset_category):
        # print("acc class[{}]:{}".format(i, _mlp.evaluate(test_data[i], test_target[i])[1]))
    ### 各層出力を可視化


def concate_elements(_list):
    concated = []
    for i in _list:
        concated += i
    return concated

def calculate_tree_shape(input_size, output_size=1):
    import math
    shape = [input_size]
    while shape[-1] > 1:
        shape.append(math.ceil(shape[-1]/2))
    return [shape[i]*(output_size if i != 0 else 1) for i in range(len(shape))]
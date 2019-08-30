from _model_weightGAN import *
import cv2
from time import *
import datetime
from draw_architecture import *
from _tree_main_mnist import divide_data
import copy

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


def visualize_network(weights, acc, comment="", non_active_neurons=None):
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

def sort_weights(_weights, target_layer=-1):
    weights = [_weights[2*i] for i in range(len(_weights)//2)]
    bias = [_weights[2*i+1] for i in range(len(_weights)//2)]
    sorted_weights = [np.zeros(_weights[2 * i].shape) for i in range(len(_weights) // 2)]
    sorted_bias = [np.zeros(_weights[2*i+1].shape) for i in range(len(_weights)//2)]

    # weights[i][j][k] : i層j番ノードからi+1層k番ノードへの重み結合
    for i in range(len(weights)-1):
        if target_layer >= 0:
            i = target_layer
        first_connect = [[float("inf"), None] for k in range(weights[i].shape[1])]
        for j in range(weights[i].shape[0]):
            for k in range(weights[i].shape[1]):
                if weights[i][j][k] != 0.:
                    if first_connect[k][1] is None:
                        first_connect[k] = [j, k]
        first_connect.sort()
        print("i:{} first_connect:{}".format(i, first_connect))
        ### first_connectを基にsorted_weightsを作成
        for k in range(weights[i].shape[1]):
            sorted_weights[i].T[k] = weights[i].T[first_connect[k][1]]
            sorted_bias[i][k] = bias[i][first_connect[k][1]]
            sorted_weights[i+1][k] = weights[i+1][first_connect[k][1]]
        ### weightsをソート済みweightで置換
        weights[i] = copy.deepcopy(sorted_weights[i])
        bias[i] = copy.deepcopy(sorted_bias[i])
        weights[i+1] = copy.deepcopy(sorted_weights[i+1])
        _weights = [weights[i // 2] if i % 2 == 0 else bias[i // 2] for i in range(len(_weights))]
        if target_layer >= 0:
            return _weights
    visualize_network(_weights, acc=-1, comment="sort all layer", non_active_neurons=None)
    return _weights
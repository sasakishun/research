from _model_weightGAN import *


def convert_weigths_tree2mlp(tree_weights, mlp_weights, input_size, output_size, dense_size):
    ### tree構造の重みを全結合モデルにセット
    tree_shape = tree(input_size, output_size, get_hidden_flag=True)
    print("tree_shape:{}".format(tree_shape))
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
    return mlp_weights


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
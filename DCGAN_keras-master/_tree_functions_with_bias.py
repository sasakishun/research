"""

from _model_weightGAN import *
from binary__tree_main import *

# mlpのweightにmaskをかける
def multiple_mask_to_model(_mlp, kernel_mask=None, bias_mask=None):
    _weight = get_kernel_and_bias(_mlp)# _mlp.get_weights()
    _mlp.summary()
    print("_weight in multiple_mask_to_model:{}".format([np.shape(i) for i in _weight]))
    if kernel_mask is not None:
        for i in range(len(_weight) // 2):
            print("multiple num :{}".format(i))
            _weight[i * 2] *= kernel_mask[i]
    if bias_mask is not None:
        for i in range(len(_weight) // 2):
            print("multiple num :{}".format(i))
            _weight[i * 2 + 1] *= bias_mask[i]
    # print("_weight in multiple_mask_to_model")
    # for i, weight in enumerate(_weight):
        # print("_weight[{}]:{}".format(i, weight))
    _mlp = set_weights(_mlp, _weight) # _mlp.set_weights(_weight)
    return _mlp

def batchNormalization_is_used(_weights):
    if np.shape(_weights[0]) == np.shape(_weights[1]):
        return True
    else:
        return False

# mlpのmaskを再計算・更新
def update_mask_of_model(_mlp):
    kernel_mask, bias_mask = get_kernel_bias_mask(_mlp)  # mask取得
    _weight = _mlp.get_weights()
    _mlp = myMLP(get_layer_size_from_weight(_weight), kernel_mask=kernel_mask,
                 bias_mask=bias_mask)  # mask付きモデル宣言
    _mlp.set_weights(_weight)  # 学習済みモデルの重みセット
    return _mlp

# 入力重み_weightsからNN構造(各層のノード数)を返す
def get_layer_size_from_weight(_weights=None):
    if _weights is None:
        d = np.load(cf.Save_np_mlp_path)
        print("np.load(cf.Save_np_mlp_path):{}".format(d))
        return get_layer_size_from_weight(np.load(cf.Save_np_mlp_path))
    else:
        print("_weights:{}".format(_weights))
        return [np.shape(_weights[0])[0]] + [np.shape(i)[1] for i in _weights if i.ndim==2]

# プルーニングしmaskも更新
def prune_and_update_mask(_mlp, X_data, y_data):
    _mlp = _weight_pruning(_mlp, X_data, y_data)  # pruning重み取得
    _mlp = update_mask_of_model(_mlp)
    return _mlp

# mlpモデル or weightsリストから、kernelとバイアス(BNパラメータ抜き)を返す
def get_kernel_and_bias(_mlp):
    _weights = model2weights(_mlp)
    if batchNormalization_is_used(_weights):
        div, remain = get_kernel_start_index_and_set_size(_mlp)
        kernel_and_bias = []
        for i in range(len(_weights) // div):
            kernel_and_bias.append(_weights[i * div + remain])
            kernel_and_bias.append(_weights[i * div + remain + 1])
        return kernel_and_bias
    else:
        return _weights

def get_kernel_start_index_and_set_size(_mlp):
    kernel_start = 0
    set_size = len(get_kernel_and_bias(_mlp))//2
    _weights = _mlp.get_weights()
    for i, _weight in enumerate(_weights):
        if _weight.ndim == 2:
            kernel_start = i
            break
    return kernel_start, set_size

# _mlpモデルに重みをセット(BNパラメータ有り無し両対応)
def set_weights(_mlp, _weights):
    kernel_and_bias = _mlp.get_weights()
    div, remain = get_kernel_start_index_and_set_size(_mlp)
    # set元でBN使用
    if batchNormalization_is_used(_weights):
        # print("set元でBN使用")
        if batchNormalization_is_used(_mlp.get_weights):
            # print("set先でBN使用")
            _mlp.set_weights(_weights)
        else:
            # print("set先でBNなし")
            for i in range(len(_weights) // div):
                kernel_and_bias[2 * i] = _weights[i * div + remain]
                kernel_and_bias[2 * i+1] = _weights[i * div + remain + 1]
            _mlp.set_weigths(kernel_and_bias)
    else:
        # print("set元でBNなし")
        if batchNormalization_is_used(_mlp.get_weights()):
            # print("set先でBN使用")
            for i in range(len(_weights) // 2):
                # print("kernel_and_bias[{}]:{} = _weights[{}]:{}".format(
                    # i * div + remain, kernel_and_bias[i * div + remain], 2*i, _weights[2 * i]))
                # print("kernel_and_bias[{}]:{} = _weights[{}]:{}".format(
                    # i * div + remain + 1, kernel_and_bias[i * div + remain + 1], 2*i+1, _weights[2 * i + 1]))
                kernel_and_bias[i * div + remain] = _weights[2 * i]
                kernel_and_bias[i * div + remain + 1] = _weights[2 * i+1]
            _mlp.set_weights(kernel_and_bias)
        else:
            # print("set先でBNなし")
            _mlp.set_weights(get_kernel_and_bias(_weights))
    return _mlp

# 入力:mlpオブジェクト->重みを返す,入力:weightsリスト->そのまま返す
def model2weights(_mlp):
    if str(type(_mlp)) == "<class 'keras.engine.training.Model'>":
        return _mlp.get_weights()
    elif str(type(_mlp)) != "list":
        return _mlp
    else:
        print("Error in model2weight : input_type must be Model or weight_list")
        exit()

# weightsのうちkernelとbiasを分けて返す
def separate_kernel_and_bias(weights):
    return [weights[i] for i in range(len(weights)) if i % 2 == 0], \
           [weights[i] for i in range(len(weights)) if i % 2 == 1]

# mlpオブジェクトor重みリストからkernel_maskとbiasマスクを返す
def get_kernel_bias_mask(_mlp):
    weights = get_kernel_and_bias(_mlp)
    # 入力 : np.arrayのリスト　ex) [(13, 5), (5,), (5, 4), (4,), (4, 2), (2,), (2, 3), (3,)]
    return separate_kernel_and_bias([np.where(weight != 0, 1, 0) for weight in weights])

def load_weights_and_generate_mlp():
    _mlp = myMLP(get_layer_size_from_weight())
    _mlp.load_weights(cf.Save_mlp_path)
    return _mlp
"""
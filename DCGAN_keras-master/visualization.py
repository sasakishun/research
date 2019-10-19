import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick  # 目盛り操作に必要なライブラリを読み込みます
import pylab
from datetime import datetime
import config_mnist as cf
import cv2


def visualize(x, y, labels, ite, testflag, showflag=False, comment="", y_range=None, correct=None, incorrect=None,
              save_fig=True, get_each_color=False, layer_type=None):
    # x : [[クラス0の訓練データ], [クラス1..]...., []]
    """
    _max_list_size = 0
    xtick_flag = False
    for i in range(len(x)):
        _max_list_size = max(_max_list_size, len(x[0]))
        if (not xtick_flag) and _max_list_size > 0:
            plt.xticks(range(len(x[i][0])))
            xtick_flag = True
    # _max_list_size = min(100, _max_list_size)
    """

    # plt.figure(figsize=(_max_list_size // 2 + 5, _max_list_size // 2 + 5), dpi=100)
    plt.figure(figsize=(8, 8), dpi=100)
    # colors = ["tomato", "black", "lightgreen"]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
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
    input_size = len(x[0][0])
    class_num = len(x)//2
    # プロット
    # print("x:{}".format(len(x)))
    # for i in range(len(x)):
    # print("x[{}]:{}".format(i, np.shape(x[i])))
    correct_range = [[[0., 0.] for _ in range(len(x[i][0]))] for i in range(len(x) // 2)]
    for i in range(len(x) // 2):
        # print("len(x[{}]):{}".format(i, len(x[i])))
        if len(x[i]) > 0:
            correct_range[i] = [[float("{}".format(min([x[i][k][j] for k in range(len(x[i]))]))),
                                 float("{}".format(max([x[i][k][j] for k in range(len(x[i]))])))]
                                for j in range(len(x[i][0]))]
            # print("correct_range[{}]:{}".format(i, correct_range[i]))

    # out_of_rangeのshape=(クラス数, ノード数) -> 中身: 間違いノード番号のリスト
    out_of_range = [[[] for _ in range(input_size)] for _ in range(len(labels) // 2)]
    out_of_range_with_color = [[[] for _ in range(input_size)] for _ in range(len(labels) // 2)]

    # out_of_range = [[[[0番ノードミスサンプル番号=2, 3, 4], [1番ノードミス]], [], [] ], [], []]
    for i in range(len(x)):  # 3クラス分類->正解*3,不正解*3でi < 6
        # print("x[{}]:{} --------".format(i, np.shape(x[i])))
        if len(x[i]) > 0:  # 各サンプルをプロット
            # print("x[{}]:{} len:{}".format(i, x[i], len(x[i])))
            # for j in range(min(500, input_size)):
            for j in range(input_size):  # j列目(13次元入力ならj<13)==参照ノード番号
                ### 正解入力をプロット
                if i < len(labels) // 2:
                    _x = [j + 0.04 * (i - len(labels) // 2) for _ in range(len(x[i]))]
                    _y = np.array(x[i])[:, j]
                    plt.scatter(_x, _y, color=colors[i], label=(i if not labels else labels[i]) if j == 0 else None,
                                marker=".")
                ### 正解入力をプロット

                ### 不正解入力をプロット
                else:
                    _x = [j + 0.04 * (i - len(labels) // 2 + 1) for _ in range(len(x[i]))]  # j番目サンプルの横軸座標
                    _y = np.array(x[i])[:, j]  # j番目のサンプルの縦軸座標
                    _in = [[], []]
                    _out = [[], []]
                    # print("_y:{}".format(_y))
                    for k in range(len(_x)):
                        ### 正解範囲に入っていたらo,それ以外はx
                        # print("_x:{}".format(_x))
                        # print("_y[k={}]:{}".format(k, _y[k]))
                        # print("correct_range[{}][{}]:{}".format(i%(len(labels)//2), int(_x[k]), correct_range[i%(len(labels)//2)][int(_x[k])]))

                        # 正解範囲外=「(-∞, ∞)をクラス数で分割」->要実装10/5～
                        # 訓練データの出力範囲内に収まったクラス
                        mergin = {"under": -0.01, "over": 0.01}
                        if (correct_range[i % (len(labels) // 2)][int(_x[k])][0] + mergin["under"] < _y[k]) \
                                and (
                                    _y[k] < mergin["over"] + correct_range[i % (len(labels) // 2)][int(_x[k])][1]):  # \
                            # and _y[k] > 0:# そのクラスの分類に不要ノードはrelu出力=0に集中するため
                            _in[0].append(_x[k])
                            _in[1].append(_y[k])
                        else:  # 訓練データの出力範囲外となったクラス
                            _out[0].append(_x[k])
                            _out[1].append(_y[k])
                            plt.annotate(k, (_x[k], _y[k]), size=10)
                            out_of_range[i % (len(labels) // 2)][int(_x[k])]\
                                .append({"sample": k, "color": colors[i % (len(labels) // 2)], "value":_y[k]})
                            # out_of_range[i%(len(labels)//2)].append(k)
                        # plt.annotate(k, (_x[k], _y[k]), size=10)
                        ### 正解範囲に入っていたらo,それ以外はx

                        # どの分類クラスに入っているか、色を指定して返す
                        # i % (len(labels) // 2) = 現在参照サンプルのクラス番号
                        target_class = i % (len(labels) // 2)
                        if get_each_color:
                            # print("_class:{} _node:{} _sample:{}".format(i % (len(labels) // 2), j, k))
                            if layer_type == "output": # 出力層の場合
                                # 参照サンプルがsoftmax最大値ノードである場合
                                if np.argmax(np.array([x[i][k][_node] for _node in range(input_size)])) == j:
                                    # 参照サンプルが間違い出力である場合
                                    # j: 現在参照しているノード番号
                                    if j != target_class:
                                        out_of_range_with_color[target_class][j].append(
                                            {"sample": k, "color": colors[j], "value": _y[k],
                                             "correct_range":correct_range[target_class][j]})
                                        # print("output:{} i % (len(labels) // 2):{}"
                                              # .format(np.array([x[i][k][_node] for _node in range(input_size)]), i % (len(labels) // 2)))
                                    else:
                                        out_of_range_with_color[target_class][j].append(
                                            {"sample": k, "color": "white", "value": _y[k],
                                             "correct_range":correct_range[target_class][int(_x[k])]})
                                else:
                                    out_of_range_with_color[target_class][j].append(
                                        {"sample": k, "color": "white", "value": _y[k],
                                             "correct_range":correct_range[target_class][j]})
                            elif layer_type == "input":  # 入力層の場合
                                belong_no_class = True
                                # 異常ノードである場合
                                if not((correct_range[target_class][int(_x[k])][0] + mergin["under"] < _y[k])\
                                               and (_y[k] < mergin["over"] + correct_range[target_class][int(_x[k])][1])):
                                    for _class in range(class_num):
                                        if (correct_range[_class][int(_x[k])][0] + mergin["under"] < _y[k]) \
                                                and (_y[k] < mergin["over"] + correct_range[_class][int(_x[k])][1]):
                                            out_of_range_with_color[target_class][int(_x[k])].append(
                                                {"sample": k, "color": colors[_class], "value": _y[k],
                                             "correct_range":correct_range[target_class][int(_x[k])]})
                                            belong_no_class = False
                                    if belong_no_class:  # どのクラス分布にも当てはまらない場合
                                        out_of_range_with_color[target_class][int(_x[k])].append(
                                            {"sample": k, "color": "gray", "value": _y[k],
                                             "correct_range":correct_range[target_class][int(_x[k])]})
                                        # print("gray used _class:{} _node:{}".format(i % (len(labels) // 2), int(_x[k])))
                                # 正常ノードである場合
                                else:
                                    out_of_range_with_color[target_class][int(_x[k])].append(
                                        {"sample": k, "color": "white", "value": _y[k],
                                             "correct_range":correct_range[target_class][int(_x[k])]})
                            else: # 中間層の場合
                                belong_no_class = True
                                # 異常ノードである場合
                                if not((correct_range[target_class][int(_x[k])][0] + mergin["under"] < _y[k])\
                                               and (_y[k] < mergin["over"] + correct_range[target_class][int(_x[k])][1])):
                                    for _class in range(class_num):
                                        if (correct_range[_class][int(_x[k])][0] + mergin["under"] < _y[k]) \
                                                and (_y[k] < mergin["over"] + correct_range[_class][int(_x[k])][1]):
                                            out_of_range_with_color[target_class][int(_x[k])].append(
                                                {"sample": k, "color": colors[_class], "value": _y[k],
                                             "correct_range":correct_range[target_class][int(_x[k])]})
                                            belong_no_class = False
                                    if belong_no_class:  # どのクラス分布にも当てはまらない場合
                                        # if _y[k] < mergin["under"] or mergin["over"] < _y[k]: # 出力!=0
                                        out_of_range_with_color[target_class][int(_x[k])].append(
                                            {"sample": k, "color": "gray", "value": _y[k],
                                             "correct_range":correct_range[target_class][int(_x[k])]})
                                        # print("gray used _class:{} _node:{}".format(i % (len(labels) // 2), int(_x[k])))
                                        # else:# 出力≒0
                                            # out_of_range_with_color[i % (len(labels) // 2)][int(_x[k])].append({"sample": k, "color": "white", "value": _y[k]})
                                            # print("white used _class:{} _node:{}".format(i % (len(labels) // 2), int(_x[k])))
                                else:
                                    out_of_range_with_color[target_class][int(_x[k])].append(
                                        {"sample": k, "color": "white", "value": _y[k],
                                         "correct_range":correct_range[target_class][int(_x[k])]})
                        # 単純に各ノードが収まる訓練正解分布の色で着色
                        """
                        if get_each_color:
                            print("_class:{} _node:{} _sample:{}".format(i % (len(labels) // 2), int(_x[k]), k))
                            if layer_type == "output": # 出力層の場合
                                if np.argmax(np.array([x[i][k][_node] for _node in range(input_size)])) == j:
                                    out_of_range_with_color[i % (len(labels) // 2)][int(_x[k])].append(
                                        {"sample": k, "color": colors[j], "value": _y[k]})
                                    print("output:{} i % (len(labels) // 2):{}"
                                          .format(np.array([x[i][k][_node] for _node in range(input_size)]), i % (len(labels) // 2)))
                                else:
                                    out_of_range_with_color[i % (len(labels) // 2)][int(_x[k])].append(
                                        {"sample": k, "color": "white", "value": _y[k]})
                            elif layer_type == "input":  # 入力層の場合
                                belong_no_class = True
                                for _class in range(len(correct_range)):
                                    if (correct_range[_class][int(_x[k])][0] + mergin["under"] < _y[k]) \
                                            and (_y[k] < mergin["over"] + correct_range[_class][int(_x[k])][1]):
                                        out_of_range_with_color[i % (len(labels) // 2)][int(_x[k])].append(
                                            {"sample": k, "color": colors[_class], "value": _y[k]})
                                        belong_no_class = False
                                if belong_no_class:  # どのクラス分布にも当てはまらない場合
                                    out_of_range_with_color[i % (len(labels) // 2)][int(_x[k])].append(
                                        {"sample": k, "color": "gray", "value": _y[k]})
                                    print("gray used _class:{} _node:{}".format(i % (len(labels) // 2), int(_x[k])))
                            else: # 中間層の場合
                                belong_no_class = True
                                for _class in range(len(correct_range)):
                                    if (correct_range[_class][int(_x[k])][0] + mergin["under"] < _y[k]) \
                                            and (_y[k] < mergin["over"] + correct_range[_class][int(_x[k])][1]):
                                        # if  _y[k] > mergin["over"]: # 0.1以上の場合のみ分布範囲内とみなす
                                        out_of_range_with_color[i % (len(labels) // 2)][int(_x[k])].append(
                                            {"sample": k, "color": colors[_class], "value": _y[k]})
                                        belong_no_class = False
                                if belong_no_class: # どのクラス分布にも当てはまらない場合
                                    if _y[k] > mergin["over"]: # 出力0でない
                                        out_of_range_with_color[i % (len(labels) // 2)][int(_x[k])].append({"sample": k, "color": "gray", "value": _y[k]})
                                        print("gray used _class:{} _node:{}".format(i % (len(labels) // 2), int(_x[k])))
                                    else:# 出力≒0
                                        out_of_range_with_color[i % (len(labels) // 2)][int(_x[k])].append({"sample": k, "color": "white", "value": _y[k]})
                                        print("white used _class:{} _node:{}".format(i % (len(labels) // 2), int(_x[k])))
                            """
                    plt.scatter(_in[0], _in[1], color=colors[i % (len(labels) // 2)],
                                label=(i if not labels else labels[i]) if j == 0 else None, marker=".")
                    plt.scatter(_out[0], _out[1], color=colors[i % (len(labels) // 2)],
                                label=(i if not labels else labels[i]) if j == 0 else None, marker="x")
                    ### 不正解入力をプロット

    correct_range_str = ""
    print("out_of_range:{}\n".format(out_of_range))
    for i in range(len(out_of_range)):
        print("out_of_range[{}]:{}".format(i, [[k["sample"] for k in j] for j in out_of_range[i]]))
        correct_range_str += "out_of_range[{}]:{}\n".format(i, set(concate_elements([[k["sample"] for k in j] for j in out_of_range[i]])))
    # for i in range(len(correct_range)):
    # "correct_range[{}]:{}".format(i, correct_range[i]) + "\n"
    plt.xlabel("{} node\n{}".format(comment, correct_range_str))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
    plt.title("ite:{} {}".format(ite, "test" if testflag else "train"))
    if not testflag:
        plt.ylabel("output")
    # y軸に1刻みにで小目盛り(minor locator)表示
    plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(1))
    plt.gca().xaxis.set_major_locator(tick.MultipleLocator(1))
    # 小目盛りに対してグリッド表示
    plt.grid(which='minor')
    # 右側の余白を調整
    pylab.subplots_adjust(right=0.7)

    if save_fig:
        # save as png
        import os
        path = os.getcwd() + r"\visualized_iris"
        print("saved to -> " + path + "\{}{}".format("test" if testflag else "train", ite))
        if ite % cf.Iteration == 0:
            plt.savefig(path + "{}{}_{}".format(r"\test\test" if testflag else r"\train\train", ite,
                                                datetime.now().strftime("%Y%m%d%H%M%S")))
        else:
            plt.savefig(path + "{}{}".format(r"\test\test" if testflag else r"\train\train", ite))
    plt.close()
    if get_each_color:
        return out_of_range_with_color
    else:
        return out_of_range

    if showflag:
        path = r"C:\Users\papap\Documents\research\DCGAN_keras-master\visualized_iris\network_architecture"
        import os
        if not os.path.exists(path):  # ディレクトリがないとき新規作成
            path = r"C:\Users\xeno\Documents\research\DCGAN_keras-master\visualized_iris\network_architecture"
            os.makedirs(path + r"\test")
            os.makedirs(path + r"\test")
        path += "{}{}_{}".format(
            r"\test" if testflag else r"\train",
            r"\{}".format(datetime.now().strftime("%Y%m%d%H%M%S")),
            "test" if testflag else "train")
        plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
        # plt.show()
    plt.close()
    path += ".png"
    print("path_real:{}".format(path))
    img = cv2.imread(path)
    print("\n\nimg:{}\n\n".format(type(img)))
    return img


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    # print("im_list[0]:{}".format(im_list[0]))
    h_min = max(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = max(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def concate_elements(_list):
    import copy
    concated = copy.deepcopy(_list)
    for i in concated[1:]:
        concated[0] += i
    return concated[0]

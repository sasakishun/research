from binary__tree_main import *


# 入力: model, クラスごとに分かれた正解訓練データ
# 出力: 正解になるが不適切な入力データ
# 正解訓練データをつぎはぎすることで入力空間では正しい分布
# ->つぎはぎ合成画像がきれいすぎて、分布外になりにくい
def get_adversarial_example(model, correct_inputs, img_shape, correct_ranges=None, test=False, save_img=False,
                            random_flag=False):
    if correct_ranges is None:
        correct_ranges = get_correct_ranges_from_data(model, correct_inputs)
    shape = np.shape(model.get_weights()[0])[0]
    # print("shape:{}".format(shape))
    # print(np.random.rand(shape))
    # print("\n\n\n\n\n")
    # for i in correct_inputs:
    # print(i)
    # print("\n\n\n\n\n")
    model_size = get_layer_size_from_weight(model.get_weights())
    if correct_inputs is not None:
        datas = [[] for _ in range(model_size[-1])]
        # correctに収まるノイズ画像を作成し,datasに追加
        # 訓練データの一部を合成すればいい
        if random_flag:  # model_size[0] == 784:
            mergin = 100  # 0.05
            _min = [min(correct_ranges, key=lambda x: x[0][i][0])[0][i][0] for i in range(model_size[0])]
            _max = [max(correct_ranges, key=lambda x: x[0][i][1])[0][i][1] for i in range(model_size[0])]
            for j in range(1000):
                if j % 100 == 0:
                    print("{} sample generated".format(j))
                data = [0 for _ in range(model_size[0])]  # np.random.rand(shape)
                for i in range(model_size[0]):
                    # _min = min(correct_ranges, key=lambda x: x[0][i][0])[0][i][0]
                    # _max = max(correct_ranges, key=lambda x: x[0][i][1])[0][i][1]
                    data[i] = random.choice([random.uniform(_min[i], min(_max[i], _min[i] + mergin)),
                                             random.uniform(max(_min[i], _max[i] - mergin), _max[i])])
                datas[np.argmax(feed_forward(model, [[data]])[-1])].append(data)
            return datas
        for _class in range(len(correct_inputs)):
            j = 0
            print("    generating class[{}]".format(_class))
            mergin = 0.05
            while j < 100:
                # mergin += 0.002
                # for j in range(100):# len(correct_inputs[_class])):
                data = []
                for i in range(model_size[0]):
                    _min = min(correct_inputs[_class], key=lambda x: x[i])[i]
                    _max = max(correct_inputs[_class], key=lambda x: x[i])[i]
                    # print("min:{}".format(_min))
                    # print("max:{}".format(_max))
                    data.append(random.choice([random.uniform(_min, min(_max, _min + mergin)),
                                               random.uniform(max(_min, _max - mergin), _max)]))
                    # data.append(sorted(correct_inputs[_class], key=lambda x:x[i])[j][i])
                    # data.append(random.choice(correct_inputs[_class])[i])
                if np.argmax(feed_forward(model, [[data]], target=None)[-1]) == _class:
                    if test:
                        out_of_range_num = adversarial_test(model, data, correct_ranges[_class])
                    else:
                        out_of_range_num = None
                    datas[_class].append(np.array(data))
                    j += 1
                    if j % 10 == 0:
                        print("class:{}_{}".format(_class, j))
                    if save_img:
                        SaveImgFromList([data, data],
                                        [img_shape[0], img_shape[1]],
                                        tag=["{}".format(_class), "{}".format(out_of_range_num)],
                                        output=[feed_forward(model, [[data]], target=None)[-1],
                                                feed_forward(model, [[data]], target=None)[-1]],
                                        comment="{}".format(j))()

            """
            if np.argmax(feed_forward(model, [[data]], target=None)[-1]) == target:
                print(feed_forward(model, [[data]], target=None)[-1])
                SaveImgFromList([data, data],
                                [8, 8],
                                tag=["{}".format(target), "{}".format(target)],
                                output=[feed_forward(model, [[data]], target=None)[-1],
                                        feed_forward(model, [[data]], target=None)[-1]])()
                return data
            """
        print("generated adversarial example:{}".format([len(i) for i in datas]))
        return datas
    else:
        data = []
        target = []
        print("generating advesarial example")
        for _ in range(1000):
            _data = np.random.rand(shape)
            data.append(_data)
            target.append(np.eye(model_size[-1])[np.argmax(feed_forward(model, [[_data]], target=None)[-1])])
        return data, target


# 入力: クラス分けされた訓練データ(クラス数,サンプル数)
def get_correct_ranges_from_data(model, data, get_pdfs=False):
    model_size = get_layer_size_from_weight(model.get_weights())
    if get_pdfs:
        each_node_outs = [[[[] for _ in range(node_num)] for node_num in model_size] for _ in range(model_size[-1])]

    # (クラス, 層, ノード, 正解範囲)
    correct_ranges = [[[[float("inf"), -float("inf")] for _ in range(node_num)] for node_num in model_size] for _ in
                      range(model_size[-1])]
    print("get_correct_ranges_from_data.....")
    for _class in range(model_size[-1]):
        print("class[{}]".format(_class))
        for _sample in data[_class]:
            hidden_out = [i[0][0] for i in feed_forward(model, [[_sample]], target=None)]
            for _layer in range(len(model_size)):
                for _node in range(model_size[_layer]):
                    correct_ranges[_class][_layer][_node][0] = min(correct_ranges[_class][_layer][_node][0],
                                                                   hidden_out[_layer][_node])
                    correct_ranges[_class][_layer][_node][1] = max(correct_ranges[_class][_layer][_node][1],
                                                                   hidden_out[_layer][_node])
                    if get_pdfs:
                        each_node_outs[_class][_layer][_node].append(hidden_out[_layer][_node])
    if get_pdfs:
        from statistics import mean, median, variance, stdev
        _mean = [[[mean(each_node_outs[_class][_layer][_node]) for _node in range(node_num)] for
                  _layer, node_num in enumerate(model_size)] for _class in range(model_size[-1])]
        _stdev = [[[stdev(each_node_outs[_class][_layer][_node]) for _node in range(node_num)] for
                   _layer, node_num in enumerate(model_size)] for _class in range(model_size[-1])]
        """
        # 正規分布における、「正常範囲境界」の「小さい方の出力」
        _edge_output = [[[stdev(each_node_outs[_class][_layer][_node]) for _node in range(node_num)] for
                   _layer, node_num in enumerate(model_size)] for _class in range(model_size[-1])]
       """

        if False:
            for _class in range(model_size[-1]):
                for _layer in range(len(model_size)):
                    for _node in range(model_size[_layer]):
                        print("class:{} layer:{} node:{}\ncorrect_range:{}\nmean:{:.2f}\nstdev:{:.2f}\n"
                              .format(_class, _layer, _node, correct_ranges[_class][_layer][_node],
                                      mean(each_node_outs[_class][_layer][_node]),
                                      stdev(each_node_outs[_class][_layer][_node])))
        pdfs = PDFs(correct_ranges=correct_ranges, mean=_mean, stdev=_stdev)
        """
        pdfs = [[[lambda x: (pdf_output(x,
                                        mean(each_node_outs[_class][_layer][_node]),
                                        stdev(each_node_outs[_class][_layer][_node]))
                             if correct_ranges[_class][_layer][_node][0] <= x <= correct_ranges[_class][_layer][_node][
            1]
                             else 0)
                  for _node in range(model_size[_layer])]
                 for _layer in range(len(model_size))]
                for _class in range(model_size[-1])]
        print("correct_ranges\n{}".format(correct_ranges))
        print("\ncorrect_range\n{}".format(correct_ranges[0][0][0][0]))
        for _class in range(model_size[-1]):
            for _layer in range(len(model_size)):
                for _node in range(model_size[_layer]):
                    print("class:{} layer:{} node:{}\ncorrect_range:{}\nlambda:{}\n"
                          .format(_class, _layer, _node, correct_ranges[_class][_layer][_node],
                                  pdfs[_class][_layer][_node]))
                    print("\npdfs\n{}".format(pdfs[_class][_layer][_node](0)))
        """
        for i in range(100):
            print("pdf({}):{}".format(i / 100, pdfs(0, 0, 0, correct_ranges[0][0][0][0] + i / 100)))
        return correct_ranges, pdfs
    else:
        return correct_ranges


def pdf_output(x, mean, stdev):
    import numpy as np
    return np.exp(-((x - mean) ** 2) / (2 * stdev ** 2))  # / np.sqrt(2 * np.pi * (stdev ** 2))


class PDFs:
    def __init__(self, correct_ranges, mean, stdev):
        self.correct_ranges = correct_ranges
        self.mean = mean
        self.stdev = stdev

    def __call__(self, _class, _layer, _node, x, *args, **kwargs):
        return pdf_output(x, self.mean[_class][_layer][_node], self.stdev[_class][_layer][_node]) \
            if self.correct_ranges[_class][_layer][_node][0] <= x \
               <= self.correct_ranges[_class][_layer][_node][1] else 0

    def means(self):
        return self.mean

    def stdev(self):
        return self.stdev

if __name__ == '__main__':
    for i in range(1000):
        print("{}: {}".format(i / 100, pdf_output(i / 100, 0, 1)))
    exit()
    _mlp = load_weights_and_generate_mlp()
    adversarial_example = get_adversarial_example(_mlp, 0)

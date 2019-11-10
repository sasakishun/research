from binary__tree_main import *

# 入力: model, クラスごとに分かれた正解訓練データ
# 出力: 正解になるが不適切な入力データ
# 正解訓練データをつぎはぎすることで入力空間では正しい分布
# ->つぎはぎ合成画像がきれいすぎて、分布外になりにくい
def get_adversarial_example(model, correct_inputs, img_shape, correct_ranges=None, test=False, save_img=False, random_flag=False):
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
    if correct_inputs is not  None:
        datas = [[] for _ in range(model_size[-1])]
        # correctに収まるノイズ画像を作成し,datasに追加
        # 訓練データの一部を合成すればいい
        if random_flag:# model_size[0] == 784:
            mergin = 100 # 0.05
            _min = [min(correct_ranges, key=lambda x: x[0][i][0])[0][i][0] for i in range(model_size[0])]
            _max = [max(correct_ranges, key=lambda x: x[0][i][1])[0][i][1] for i in range(model_size[0])]
            for j in range(1000):
                if j % 100 == 0:
                    print("{} sample generated".format(j))
                data = [0 for _ in range(model_size[0])]# np.random.rand(shape)
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
                    _min = min(correct_inputs[_class], key=lambda x:x[i])[i]
                    _max = max(correct_inputs[_class], key=lambda x:x[i])[i]
                    # print("min:{}".format(_min))
                    # print("max:{}".format(_max))
                    data.append(random.choice([random.uniform(_min, min(_max, _min+mergin)),
                                               random.uniform(max(_min, _max-mergin), _max)]))
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
def get_correct_ranges_from_data(model, data):
    model_size = get_layer_size_from_weight(model.get_weights())
    # (クラス, 層, ノード, 正解範囲)
    correct_ranges = [[[[float("inf"), -float("inf")] for _ in range(layer_num)] for layer_num in model_size] for _ in range(model_size[-1])]
    print("get_correct_ranges_from_data.....")
    for _class in range(model_size[-1]):
        print("class[{}]".format(_class))
        for _sample in data[_class]:
            hidden_out = [i[0][0] for i in feed_forward(model, [[_sample]], target=None)]
            for _layer in range(len(model_size)):
                for _node in range(model_size[_layer]):
                    # print("correct_range:{}".format(correct_ranges[_class][_layer][_node]))
                    # print("hidden_out[_layer][_node]:{}".format(hidden_out[_layer][_node]))
                    correct_ranges[_class][_layer][_node][0] = min(correct_ranges[_class][_layer][_node][0], hidden_out[_layer][_node])
                    correct_ranges[_class][_layer][_node][1] = max(correct_ranges[_class][_layer][_node][1], hidden_out[_layer][_node])
    return correct_ranges

if __name__ == '__main__':
    _mlp = load_weights_and_generate_mlp()
    adversarial_example = get_adversarial_example(_mlp, 0)
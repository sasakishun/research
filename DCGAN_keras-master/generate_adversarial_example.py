from binary__tree_main import *

# 入力: model, クラスごとに分かれた正解訓練データ
# 出力: 正解になるが不適切な入力データ
# 正解訓練データをつぎはぎすることで入力空間では正しい分布
# ->つぎはぎ合成画像がきれいすぎて、分布外になりにくい
def get_adversarial_example(model, correct_inputs, img_shape):
    shape = np.shape(model.get_weights()[0])[0]
    print("shape:{}".format(shape))
    print(np.random.rand(shape))
    print("\n\n\n\n\n")
    for i in correct_inputs:
        print(i)
    print("\n\n\n\n\n")
    model_size = get_layer_size_from_weight(model.get_weights())
    if correct_inputs is not  None:
        datas = [[] for _ in range(len(correct_inputs))]
        # correctに収まるノイズ画像を作成し,datasに追加
        # 訓練データの一部を合成すればいい
        for _class in range(len(correct_inputs)):
            for _ in range(100):
                data = []
                for i in range(model_size[0]):
                    data.append(random.choice(correct_inputs[_class])[i])
                if np.argmax(feed_forward(model, [[data]], target=None)[-1]) == _class:
                    datas[_class].append(np.array(data))
                    SaveImgFromList([data, data],
                                    [img_shape[0], img_shape[1]],
                                    tag=["{}".format(_class), "{}".format(_class)],
                                    output=[feed_forward(model, [[data]], target=None)[-1],
                                            feed_forward(model, [[data]], target=None)[-1]])()

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
        for _ in range(100):
            _data = np.random.rand(shape)
            data.append(_data)
            target.append(np.eye(model_size[-1])[np.argmax(feed_forward(model, [[_data]], target=None)[-1])])
        return data, target

if __name__ == '__main__':
    _mlp = load_weights_and_generate_mlp()
    adversarial_example = get_adversarial_example(_mlp, 0)
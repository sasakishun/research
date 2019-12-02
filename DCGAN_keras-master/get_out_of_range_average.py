import os
from matplotlib import pyplot as plt
import matplotlib.ticker as tick  # 目盛り操作に必要なライブラリを読み込みます
from statistics import mean, median, variance, stdev
from datetime import datetime
import numpy as np
import math

datasets = ["iris", "wine", "digit", "mnist"]


def reliability_test(CORRECT_TEST, MISS_TEST, show_detail=False):
    if len(MISS_TEST) == 0:
        return
    # 信頼判定テスト
    # 各ノードごとに信頼性あり無しのスレッショルド設定
    threshold = [0 for _ in range(len(CORRECT_TEST[0]))]
    for node in range(len(CORRECT_TEST[0])):
        data = [x[node] for x in CORRECT_TEST]
        _mean = mean(data)
        _median = median(data)
        _variance = variance(data)
        _stdev = stdev(data)
        threshold[node] = _mean + 2. * _stdev  # max(data)  # 3σ位置
        if show_detail:
            print("data:{}".format(data))
            print('平均: {0:.2f}'.format(_mean))
            print('中央値: {0:.2f}'.format(_median))
            print('分散: {0:.2f}'.format(_variance))
            print('標準偏差: {0:.2f}'.format(_stdev))
    adversarial_miss = adversarial_miss_num(MISS_TEST, threshold)
    correct_miss = adversarial_miss_num(CORRECT_TEST, threshold)
    print("adversarial_miss {}/{} -> {:.2f}% correct_miss {}/{} -> {:.2f}%"
          .format(adversarial_miss["total_adversarial_miss"], len(MISS_TEST),
                  100 * adversarial_miss["total_adversarial_miss"] / len(MISS_TEST),
                  correct_miss["total_adversarial_miss"], len(CORRECT_TEST),
                  100 * correct_miss["total_adversarial_miss"] / len(CORRECT_TEST)))
    return {"correct_exclude": correct_miss["total_adversarial_miss"] / len(CORRECT_TEST),
            "miss_exclude": adversarial_miss["total_adversarial_miss"] / len(MISS_TEST),
            "threshold": threshold,
            "correct_excludes": [i / len(CORRECT_TEST) for i in correct_miss["adversarial_misses"]],
            "miss_excludes": [i / len(MISS_TEST) for i in adversarial_miss["adversarial_misses"]],
            }


def adversarial_miss_num(MISS_TEST, threshold):
    adversarial_misses = [0 for _ in range(len(MISS_TEST[0]))]
    total_adversarial_miss = 0
    for miss_test in MISS_TEST:
        miss_occur = False
        for layer in range(len(miss_test)):
            if miss_test[layer] > threshold[layer]:
                adversarial_misses[layer] += 1
                miss_occur = True
        if miss_occur:
            total_adversarial_miss += 1
    return {"total_adversarial_miss": total_adversarial_miss,
            "adversarial_misses": adversarial_misses}


class Aggregate_data:
    def __init__(self):
        self.iris_miss = []
        self.iris_correct = []
        self.iris_adversarial = []
        self.iris_random_noise_exclude_rate = []
        self.iris_correct_test_exclude_rate = []
        self.iris_train_acc = []
        self.iris_test_acc = []
        self.MLP_iris_train_acc = []
        self.MLP_iris_test_acc = []

        self.wine_miss = []
        self.wine_correct = []
        self.wine_adversarial = []
        self.wine_random_noise_exclude_rate = []
        self.wine_correct_test_exclude_rate = []
        self.wine_train_acc = []
        self.wine_test_acc = []
        self.MLP_wine_train_acc = []
        self.MLP_wine_test_acc = []

        self.digit_miss = []
        self.digit_correct = []
        self.digit_adversarial = []
        self.digit_random_noise_exclude_rate = []
        self.digit_correct_test_exclude_rate = []
        self.digit_train_acc = []
        self.digit_test_acc = []
        self.MLP_digit_train_acc = []
        self.MLP_digit_test_acc = []

        self.mnist_miss = []
        self.mnist_correct = []
        self.mnist_adversarial = []
        self.mnist_random_noise_exclude_rate = []
        self.mnist_correct_test_exclude_rate = []
        self.mnist_train_acc = []
        self.mnist_test_acc = []
        self.MLP_mnist_train_acc = []
        self.MLP_mnist_test_acc = []


    def set_data(self, dataset, data=None,
                 random_noise_exclude_rate=None,
                 correct_test_exclude_rate=None,
                 train_acc=None, test_acc=None):
        if data is not None:
            eval("self." + dataset + "_miss").append(data[0])
            eval("self." + dataset + "_correct").append(data[1])
            eval("self." + dataset + "_adversarial").append(data[-1])
        if random_noise_exclude_rate is not None:
            eval("self." + dataset + "_random_noise_exclude_rate").append(random_noise_exclude_rate)
        if correct_test_exclude_rate is not None:
            eval("self." + dataset + "_correct_test_exclude_rate").append(correct_test_exclude_rate)
        if train_acc is not None:
            eval("self." + dataset + "_train_acc").append(train_acc)
        if test_acc is not None:
            eval("self." + dataset + "_test_acc").append(test_acc)

    def get_data(self):
        for _dataset in datasets:
            for _category in ["miss", "correct", "adversarial", "random_noise_exclude_rate"]:
                for i in eval("self." + _dataset + "_" + _category):
                    print(_dataset + "_" + _category + ": " + "{}".format(i))
                print("{}:{}".format("self." + _dataset + "_" + _category,
                                     len(eval("self." + _dataset + "_" + _category))))

        iris = [{"correct": [], "miss": [], "adversarial": []} for _ in range(len(self.iris_correct[0]))]
        wine = [{"correct": [], "miss": [], "adversarial": []} for _ in range(len(self.wine_correct[0]))]
        digit = [{"correct": [], "miss": [], "adversarial": []} for _ in range(len(self.digit_correct[0]))]
        mnist = [{"correct": [], "miss": [], "adversarial": []} for _ in range(len(self.mnist_correct[0]))]

        # self.に格納したリストを集計
        for _dataset in datasets:
            for _iris_correct in eval("self." + _dataset + "_correct"):
                for layer, data in enumerate(_iris_correct):
                    eval(_dataset)[layer]["correct"].append(float(data))
            for _iris_miss in eval("self." + _dataset + "_miss"):
                for layer, data in enumerate(_iris_miss):
                    eval(_dataset)[layer]["miss"].append(float(data))
            for _iris_adversarial in eval("self." + _dataset + "_adversarial"):
                for layer, data in enumerate(_iris_adversarial):
                    eval(_dataset)[layer]["adversarial"].append(float(data))

        # self.に格納したリストを平均値に変換
        for _dataset in datasets:
            for layer, data in enumerate(eval(_dataset)):
                eval(_dataset)[layer]["correct"] = sum(eval(_dataset)[layer]["correct"]) / len(
                    eval(_dataset)[layer]["correct"])
                eval(_dataset)[layer]["miss"] = sum(eval(_dataset)[layer]["miss"]) / len(eval(_dataset)[layer]["miss"])
                eval(_dataset)[layer]["adversarial"] = sum(eval(_dataset)[layer]["adversarial"]) / len(
                    eval(_dataset)[layer]["adversarial"])

        # 集計した平均をlatexで出力可能な形式に変換
        title = ["入力層    ", "第一中間層", "第二中間層", "第三中間層", "出力層    "]
        blank = " － "
        for target in [["correct", "miss"], ["correct", "adversarial"]]:
            print("\n平均異常ノード発生数:{}".format(target))
            for i in range(len(title) - 1):
                print("{}&{}/{} &{}/{} &{}/{} &{}/{} \\\\ \cline".format(
                    title[i],
                    "{:.2f}".format(iris[i][target[0]]) if i < len(iris) - 1 else blank,
                    "{:.2f}".format(iris[i][target[1]]) if i < len(iris) - 1 else blank,
                    "{:.2f}".format(wine[i][target[0]]) if i < len(wine) - 1 else blank,
                    "{:.2f}".format(wine[i][target[1]]) if i < len(wine) - 1 else blank,
                    "{:.2f}".format(digit[i][target[0]]) if i < len(digit) - 1 else blank,
                    "{:.2f}".format(digit[i][target[1]]) if i < len(digit) - 1 else blank,
                    "{:.2f}".format(mnist[i][target[0]]) if i < len(mnist) - 1 else blank,
                    "{:.2f}".format(mnist[i][target[1]]) if i < len(mnist) - 1 else blank,
                ) + r"{0-4}")
            for i in range(len(title) - 1, len(title)):
                print("{}&{}/{} &{}/{} &{}/{} &{}/{} \\\\ \Hline".format(
                    title[i],
                    "{:.2f}".format(iris[-1][target[0]]),
                    "{:.2f}".format(iris[-1][target[1]]),
                    "{:.2f}".format(wine[-1][target[0]]),
                    "{:.2f}".format(wine[-1][target[1]]),
                    "{:.2f}".format(digit[-1][target[0]]),
                    "{:.2f}".format(digit[-1][target[1]]),
                    "{:.2f}".format(mnist[-1][target[0]]),
                    "{:.2f}".format(mnist[-1][target[1]]),
                ))
            print()
        print("ノイズ排除率\n{:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\Hline".format(
            sum(self.iris_random_noise_exclude_rate) * 100 / len(self.iris_random_noise_exclude_rate),
            sum(self.wine_random_noise_exclude_rate) * 100 / len(self.wine_random_noise_exclude_rate),
            sum(self.digit_random_noise_exclude_rate) * 100 / len(self.digit_random_noise_exclude_rate),
            sum(self.mnist_random_noise_exclude_rate) * 100 / len(self.mnist_random_noise_exclude_rate),
        ))
        print()
        print("正解保持率\n{:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\Hline".format(
            100 - sum(self.iris_correct_test_exclude_rate) * 100 / len(self.iris_correct_test_exclude_rate),
            100 - sum(self.wine_correct_test_exclude_rate) * 100 / len(self.wine_correct_test_exclude_rate),
            100 - sum(self.digit_correct_test_exclude_rate) * 100 / len(self.digit_correct_test_exclude_rate),
            100 - sum(self.mnist_correct_test_exclude_rate) * 100 / len(self.mnist_correct_test_exclude_rate),
        ))

        for test_train in ["train", "test"]:
            print("提案手法の分類精度({})\n{:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\Hline".format(
                test_train,
                sum(eval("self.iris_" + test_train + "_acc")) * 100 / len(eval("self.iris_" + test_train + "_acc")),
                sum(eval("self.wine_" + test_train + "_acc")) * 100 / len(eval("self.wine_" + test_train + "_acc")),
                sum(eval("self.digit_" + test_train + "_acc")) * 100 / len(eval("self.digit_" + test_train + "_acc")),
                sum(eval("self.mnist_" + test_train + "_acc")) * 100 / len(eval("self.mnist_" + test_train + "_acc")),
            ))
            print("MLPの分類精度({})\n{:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\Hline".format(
                test_train,
                sum(eval("self.MLP_iris_" + test_train + "_acc")) * 100 / len(eval("self.MLP_iris_" + test_train + "_acc")),
                sum(eval("self.MLP_wine_" + test_train + "_acc")) * 100 / len(eval("self.MLP_wine_" + test_train + "_acc")),
                sum(eval("self.MLP_digit_" + test_train + "_acc")) * 100 / len(eval("self.MLP_digit_" + test_train + "_acc")),
                sum(eval("self.MLP_mnist_" + test_train + "_acc")) * 100 / len(eval("self.MLP_mnist_" + test_train + "_acc")),
            ))


def my_makedirs(path):
    import os
    if not os.path.isdir(path):
        os.makedirs(path)


if __name__ == '__main__':
    aggregate_data = Aggregate_data()
    # ファイルをオープンする
    dir = os.getcwd() + r"\result_NC"
    hist_dir = os.getcwd() + r"\histgram"
    files = os.listdir(dir)
    for file in files:
        # print(file)
        test_data = open(dir + r"\{}".format(file), "r")
        # 一行ずつ読み込んでは表示する
        dataset = ""
        out_of_range_average = []
        random_noise_exclude_rate = 0
        train_acc = 0
        test_acc = 0
        # 異常ノード発生数
        CORRECT_TEST = []
        MISS_TEST = []
        adversarial_example_unRandom = []
        adversarial_example_Random = []
        data_type = ""

        for i, line in enumerate(test_data):
            if i == 0:
                if line[:4] == "iris":
                    dataset = "iris"
                elif line[:4] == "wine":
                    dataset = "wine"
                elif line[:5] == "digit":
                    dataset = "digit"
                elif line[:5] == "mnist":
                    dataset = "mnist"
                elif line[:4] == "MLP_":
                    if line[4:8] == "iris":
                        dataset = "MLP_iris"
                    elif line[4:8] == "wine":
                        dataset = "MLP_wine"
                    elif line[4:9] == "digit":
                        dataset = "MLP_digit"
                    elif line[4:9] == "mnist":
                        dataset = "MLP_mnist"
                print(" dataset:{}".format(dataset))
            elif line[:20] == "out_of_range_average":
                out_of_range_average.append(eval(line[20:]))
            # 精度計算
            elif line[:16] == "advresarial_miss" and data_type[:-7] == "_Random":
                for i in range(len(line)):
                    if line[i:i + 3] == "-> ":
                        random_noise_exclude_rate = float(eval(line[i + 3:]))
            elif line[:14] == "train loss_acc":
                train_loss_acc = eval(line[14:])
                train_acc = train_loss_acc[1]
            elif line[:14] == "test  loss_acc":
                test_loss_acc = eval(line[14:])
                test_acc = test_loss_acc[1]

            # 異常ノード発生数集計
            elif line[:13] == "CORRECT_TRAIN":  # data_type -> MISS_TEST or CORRECT_TEST
                data_type = line[14:]
            elif line[:36] == "adversarial_example random_flag:True":
                data_type = line[:19] + "_Random"
            elif line[:37] == "adversarial_example random_flag:False":
                data_type = line[:19] + "_unRandom"
            elif line[0] == "[":
                # print("{} <- append :{}".format(eval(data_type), eval(line)))
                eval(data_type).append(eval(line))

        # CORRECT,MISSごとのi番目の要素を参照
        for _CORRECT_TEST, _MISS_TEST, name in \
                zip([CORRECT_TEST, CORRECT_TEST, CORRECT_TEST],
                    [MISS_TEST, adversarial_example_Random, adversarial_example_unRandom],
                    ["MISS_TEST", "RANDOM_NOISE", "RESTRICTED_RANDOM_NOISE"]):
            if len(_MISS_TEST) == 0:
                continue
            # 1ファイル終了するごとにデータ集計・統計データ算出
            _reliability = reliability_test(_CORRECT_TEST, _MISS_TEST)
            correct_exclude = _reliability["correct_exclude"]
            miss_exclude = _reliability["miss_exclude"]
            correct_excludes = _reliability["correct_excludes"]
            miss_excludes = _reliability["miss_excludes"]
            threshold = _reliability["threshold"]
            if name == "RANDOM_NOISE":
                aggregate_data.set_data(dataset, out_of_range_average, miss_exclude, correct_exclude,
                                        train_acc=train_acc, test_acc=test_acc)

            # ヒストグラム作成
            for i in range(len(_CORRECT_TEST[0])):
                _min0 = min([x[i] for x in _CORRECT_TEST])
                _max0 = max([x[i] for x in _CORRECT_TEST])
                _min1 = min([x[i] for x in _MISS_TEST])
                _max1 = max([x[i] for x in _MISS_TEST])
                _range = (min(_min0, _min1), max(_max0, _max1))
                binnum = _range[1] - _range[0] + 2
                # print("binnum: {}".format(binnum))
                plt.hist([x[i] for x in _CORRECT_TEST], label="CORRECT_TEST" + "[{:.2f}% remain ]"
                         .format((1 - correct_excludes[i]) * 100),
                         alpha=0.6, normed=True, bins=np.arange(binnum) - 0.5, align="mid")
                plt.hist([x[i] for x in _MISS_TEST], label=name + "[{:.2f}% exclude]"
                         .format(miss_excludes[i] * 100),
                         alpha=0.6, normed=True, bins=np.arange(binnum) - 0.5, align="mid")
                # 閾値による分類境界を表示
                plt.axvline(x=math.ceil(threshold[i]) - 0.5, color="red", label="μ+2σ line")
                # fig = plt.figure()
                ax = plt.subplot(111)
                # plt.yscale('log') # 縦軸をlogにできる
                ax_yticklocs = ax.yaxis.get_ticklocs()  # 目盛りの情報を取得
                ax_yticklocs = list(map(lambda x: x * len(range(binnum)) * 1.0 / binnum,
                                        ax_yticklocs))  # 元の目盛りの値にbinの幅を掛ける
                ax.yaxis.set_ticklabels(list(map(lambda x: "%0.2f" % x, ax_yticklocs)))
                # plt.ylim(0.01, )
                plt.xlim(-0.5, )
                plt.xlabel("the number of nodes outside the correct range")
                plt.ylabel("frequency")
                plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(math.ceil(binnum / 10)))
                plt.gca().xaxis.set_major_locator(tick.MultipleLocator(math.ceil(binnum / 10)))
                plt.legend()
                # plt.show()
                # exit()
                path = hist_dir + r"\{}\{}".format(dataset, name)
                if hist_dir is not None:
                    my_makedirs(path)
                plt.savefig(path + r"\{}_{}"
                            .format("layer" + str(i), datetime.now().strftime("%Y%m%d%H%M%S")))
                plt.close()
            # exit()
        if False:
            print("CORRECT_TEST:{}".format(CORRECT_TEST))
            print("MISS_TEST:{}".format(MISS_TEST))
            print("adversarial_example_unRandom:{}".format(adversarial_example_unRandom))
            print("adversarial_example_Random:{}".format(adversarial_example_Random))
            print("CORRECT_TEST:{}".format(len(CORRECT_TEST)))
            print("MISS_TEST:{}".format(len(MISS_TEST)))
            print("adversarial_example_unRandom:{}".format(len(adversarial_example_unRandom)))
            print("adversarial_example_Random:{}".format(len(adversarial_example_Random)))
            exit()
        # ファイルをクローズする
        test_data.close()
    aggregate_data.get_data()

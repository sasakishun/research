import os

datasets = ["iris", "wine", "digit", "mnist"]
class Aggregate_data:
    def __init__(self):
        self.iris_miss = []
        self.iris_correct = []
        self.iris_adversarial = []
        self.iris_ad_acc = []
        self.iris_train_acc = []
        self.iris_test_acc = []

        self.wine_miss = []
        self.wine_correct = []
        self.wine_adversarial = []
        self.wine_ad_acc = []
        self.wine_train_acc = []
        self.wine_test_acc = []

        self.digit_miss = []
        self.digit_correct = []
        self.digit_adversarial = []
        self.digit_ad_acc = []
        self.digit_train_acc = []
        self.digit_test_acc = []

        self.mnist_miss = []
        self.mnist_correct = []
        self.mnist_adversarial = []
        self.mnist_ad_acc = []
        self.mnist_train_acc = []
        self.mnist_test_acc = []

    def set_data(self, dataset, data, ad_acc, train_acc=None, test_acc = None):
        eval("self." + dataset + "_miss").append(data[0])
        eval("self." + dataset + "_correct").append(data[1])
        eval("self." + dataset + "_adversarial").append(data[-1])
        eval("self." + dataset + "_ad_acc").append(ad_acc)
        if train_acc is not None:
            eval("self." + dataset + "_train_acc").append(train_acc)
        if test_acc is not None:
            eval("self." + dataset + "_test_acc").append(test_acc)

    def get_data(self):
        for _dataset in datasets:
            for _category in ["miss", "correct", "adversarial", "ad_acc"]:
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
                eval(_dataset)[layer]["correct"] = sum(eval(_dataset)[layer]["correct"]) / len(eval(_dataset)[layer]["correct"])
                eval(_dataset)[layer]["miss"] = sum(eval(_dataset)[layer]["miss"]) / len(eval(_dataset)[layer]["miss"])
                eval(_dataset)[layer]["adversarial"] = sum(eval(_dataset)[layer]["adversarial"]) / len(eval(_dataset)[layer]["adversarial"])

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
            sum(self.iris_ad_acc) * 100 / len(self.iris_ad_acc),
            sum(self.wine_ad_acc) * 100 / len(self.wine_ad_acc),
            sum(self.digit_ad_acc) * 100 / len(self.digit_ad_acc),
            sum(self.mnist_ad_acc) * 100 / len(self.mnist_ad_acc),
        ))
        for test_train in ["train", "test"]:
            print("分類精度({})\n{:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\Hline".format(
                test_train,
                sum(eval("self.iris_" + test_train + "_acc")) * 100 / len(eval("self.iris_" + test_train + "_acc")),
                sum(eval("self.wine_" + test_train + "_acc")) * 100 / len(eval("self.wine_" + test_train + "_acc")),
                sum(eval("self.digit_" + test_train + "_acc")) * 100 / len(eval("self.digit_" + test_train + "_acc")),
                sum(eval("self.mnist_" + test_train + "_acc")) * 100 / len(eval("self.mnist_" + test_train + "_acc")),
            ))

aggregate_data = Aggregate_data()
# ファイルをオープンする
dir  = os.getcwd() + r"\result"
files = os.listdir(dir)
for file in files:
    # print(file)
    test_data = open(dir + r"\{}".format(file), "r")
    # 一行ずつ読み込んでは表示する
    dataset = ""
    out_of_range_average = []
    ad_acc = 0
    train_acc = 0
    test_acc = 0
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
            # print(line)
        if line[:20] == "out_of_range_average":
            out_of_range_average.append(eval(line[20:]))
            # print(line)
        if line[:16] == "advresarial_miss":
            for i in range(len(line)):
                if line[i:i+3] == "-> ":
                    ad_acc = float(eval(line[i+3:]))
        if line[:14] == "train loss_acc":
            train_loss_acc = eval(line[14:])
            train_acc = train_loss_acc[1]
        if line[:14] == "test  loss_acc":
            test_loss_acc = eval(line[14:])
            test_acc = test_loss_acc[1]
    # print("dataset:{}".format(dataset))
    # print(out_of_range_average)
    # print(ad_acc)
    aggregate_data.set_data(dataset, out_of_range_average, ad_acc, train_acc=train_acc, test_acc=test_acc)
    # ファイルをクローズする
    test_data.close()
aggregate_data.get_data()
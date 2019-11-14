import os

class Aggregate_data:
    def __init__(self):
        self.iris_miss = []
        self.iris_correct = []
        self.iris_adversarial = []
        self.iris_ad_acc = []

        self.wine_miss = []
        self.wine_correct = []
        self.wine_adversarial = []
        self.wine_ad_acc = []

        self.digit_miss = []
        self.digit_correct = []
        self.digit_adversarial = []
        self.digit_ad_acc = []

        self.mnist_miss = []
        self.mnist_correct = []
        self.mnist_adversarial = []
        self.mnist_ad_acc = []

    def set_data(self, dataset, data, ad_acc):
        eval("self." + dataset + "_miss").append(data[0])
        eval("self." + dataset + "_correct").append(data[1])
        eval("self." + dataset + "_adversarial").append(data[-1])
        eval("self." + dataset + "_ad_acc").append(ad_acc)

    def get_data(self):
        for _dataset in ["iris", "wine", "digit", "mnist"]:
            for _category in ["miss", "correct", "adversarial", "ad_acc"]:
                for i in eval("self." + _dataset + "_" + _category):
                    print(_dataset + "_" + _category + ": " + "{}".format(i))
                print("{}:{}".format("self." + _dataset + "_" + _category,
                                     len(eval("self." + _dataset + "_" + _category))))

        iris = [{"correct": [], "miss": [], "adversarial": []} for _ in range(len(self.iris_correct[0]))]
        wine = [{"correct": [], "miss": [], "adversarial": []} for _ in range(len(self.wine_correct[0]))]
        digit = [{"correct": [], "miss": [], "adversarial": []} for _ in range(len(self.digit_correct[0]))]
        mnist = [{"correct": [], "miss": [], "adversarial": []} for _ in range(len(self.mnist_correct[0]))]

        for _iris_correct in self.iris_correct:
            for layer, data in enumerate(_iris_correct):
                iris[layer]["correct"].append(float(data))
        for _iris_miss in self.iris_miss:
            for layer, data in enumerate(_iris_miss):
                iris[layer]["miss"].append(float(data))
        for _iris_adversarial in self.iris_adversarial:
            for layer, data in enumerate(_iris_adversarial):
                iris[layer]["adversarial"].append(float(data))

        for _wine_correct in self.wine_correct:
            for layer, data in enumerate(_wine_correct):
                wine[layer]["correct"].append(float(data))
        for _wine_miss in self.wine_miss:
            for layer, data in enumerate(_wine_miss):
                wine[layer]["miss"].append(float(data))
        for _wine_adversarial in self.wine_adversarial:
            for layer, data in enumerate(_wine_adversarial):
                wine[layer]["adversarial"].append(float(data))

        for _digit_correct in self.digit_correct:
            for layer, data in enumerate(_digit_correct):
                digit[layer]["correct"].append(float(data))
        for _digit_miss in self.digit_miss:
            for layer, data in enumerate(_digit_miss):
                digit[layer]["miss"].append(float(data))
        for _digit_adversarial in self.digit_adversarial:
            for layer, data in enumerate(_digit_adversarial):
                digit[layer]["adversarial"].append(float(data))
        for _mnist_correct in self.mnist_correct:
            for layer, data in enumerate(_mnist_correct):
                mnist[layer]["correct"].append(float(data))
        for _mnist_miss in self.mnist_miss:
            for layer, data in enumerate(_mnist_miss):
                mnist[layer]["miss"].append(float(data))
        for _mnist_adversarial in self.mnist_adversarial:
            for layer, data in enumerate(_mnist_adversarial):
                mnist[layer]["adversarial"].append(float(data))

        for _dataset in ["iris", "wine", "digit", "mnist"]:
            for layer, data in enumerate(eval(_dataset)):
                eval(_dataset)[layer]["correct"] = sum(eval(_dataset)[layer]["correct"]) / len(eval(_dataset)[layer]["correct"])
                eval(_dataset)[layer]["miss"] = sum(eval(_dataset)[layer]["miss"]) / len(eval(_dataset)[layer]["miss"])
                eval(_dataset)[layer]["adversarial"] = sum(eval(_dataset)[layer]["adversarial"]) / len(eval(_dataset)[layer]["adversarial"])

        title = ["入力層", "第一中間層", "第二中間層", "第三中間層", "出力層"]
        blank = "－"
        for target in [["correct", "miss"], ["correct", "adversarial"]]:
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
        print("{:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\Hline".format(
            sum(self.iris_ad_acc) * 100 / len(self.iris_ad_acc),
            sum(self.wine_ad_acc) * 100 / len(self.wine_ad_acc),
            sum(self.digit_ad_acc) * 100 / len(self.digit_ad_acc),
            sum(self.mnist_ad_acc) * 100 / len(self.mnist_ad_acc),
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
    # print("dataset:{}".format(dataset))
    # print(out_of_range_average)
    # print(ad_acc)
    aggregate_data.set_data(dataset, out_of_range_average, ad_acc)
    # ファイルをクローズする
    test_data.close()
aggregate_data.get_data()
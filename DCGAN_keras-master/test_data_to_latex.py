def to_latex():
    return


if __name__ == '__main__':
    iris_miss = [['0.0000', '2.0000', '2.0000'], ['0.0000', '2.0000', '2.0000'], ]
    iris_correct = [['0.1053', '0.0526', '0.0000'], ['0.1053', '0.0263', '0.0000'], ]
    iris_adversarial = [['1.5860', '0.5700', '0.1420'], ['1.6360', '0.9890', '0.0910'], ]
    iris_ad_acc = [0.5300, 0.6150, ]

    wine_miss = [['0.0000', '0.0000', '1.0000', '1.0000', '2.0000']]
    wine_correct = [['0.2778', '0.0000', '0.0000', '0.0000', '0.0000'],
                    ['0.2778', '0.0000', '0.0000', '0.0000', '0.0000']]
    wine_adversarial = [['4.2000', '0.3180', '0.2180', '0.2020', '0.1290'],
                        ['4.1160', '0.2060', '0.0210', '0.0240', '0.0150']]
    wine_ad_acc = [0.9220, 0.8440, ]

    digit_miss = [['1.1250', '3.8750', '2.1667'], ['1.3529', '5.2941', '2.1176']]
    digit_correct = [['0.0655', '0.4911', '0.0446'], ['0.0758', '0.5423', '0.0321']]
    digit_adversarial = [['12.8990', '15.2620', '0.3080'], ['13.1800', '18.9880', '0.4360']]
    digit_ad_acc = [1.0, 1.0, ]

    mnist_miss = [['0.9032', '0.6036', '1.4590'], ['0.9042', '0.7743', '1.6955']]
    mnist_correct = [['0.0786', '0.0534', '0.0013'], ['0.0649', '0.0965', '0.0026']]
    mnist_adversarial = [['199.8540', '26.2640', '0.0000'], ['151.3360', '60.4290', '0.0060']]
    mnist_ad_acc = [1.0, 1.0, ]

    iris = [{"correct": [], "miss": [], "adversarial": []} for _ in range(len(iris_correct[0]))]
    wine = [{"correct": [], "miss": [], "adversarial": []} for _ in range(len(wine_correct[0]))]
    digit = [{"correct": [], "miss": [], "adversarial": []} for _ in range(len(digit_correct[0]))]
    mnist = [{"correct": [], "miss": [], "adversarial": []} for _ in range(len(mnist_correct[0]))]

    for _iris_correct in iris_correct:
        for layer, data in enumerate(_iris_correct):
            iris[layer]["correct"].append(float(data))
    for _iris_miss in iris_miss:
        for layer, data in enumerate(_iris_miss):
            iris[layer]["miss"].append(float(data))
    for _iris_adversarial in iris_adversarial:
        for layer, data in enumerate(_iris_adversarial):
            iris[layer]["adversarial"].append(float(data))

    for _wine_correct in wine_correct:
        for layer, data in enumerate(_wine_correct):
            wine[layer]["correct"].append(float(data))
    for _wine_miss in wine_miss:
        for layer, data in enumerate(_wine_miss):
            wine[layer]["miss"].append(float(data))
    for _wine_adversarial in wine_adversarial:
        for layer, data in enumerate(_wine_adversarial):
            wine[layer]["adversarial"].append(float(data))

    for _digit_correct in digit_correct:
        for layer, data in enumerate(_digit_correct):
            digit[layer]["correct"].append(float(data))
    for _digit_miss in digit_miss:
        for layer, data in enumerate(_digit_miss):
            digit[layer]["miss"].append(float(data))
    for _digit_adversarial in digit_adversarial:
        for layer, data in enumerate(_digit_adversarial):
            digit[layer]["adversarial"].append(float(data))
    for _mnist_correct in mnist_correct:
        for layer, data in enumerate(_mnist_correct):
            mnist[layer]["correct"].append(float(data))
    for _mnist_miss in mnist_miss:
        for layer, data in enumerate(_mnist_miss):
            mnist[layer]["miss"].append(float(data))
    for _mnist_adversarial in mnist_adversarial:
        for layer, data in enumerate(_mnist_adversarial):
            mnist[layer]["adversarial"].append(float(data))

    for layer, data in enumerate(wine):
        wine[layer]["correct"] = sum(wine[layer]["correct"]) / len(wine[layer]["correct"])
        wine[layer]["miss"] = sum(wine[layer]["miss"]) / len(wine[layer]["miss"])
        wine[layer]["adversarial"] = sum(wine[layer]["adversarial"]) / len(wine[layer]["adversarial"])

    for layer, data in enumerate(iris):
        iris[layer]["correct"] = sum(iris[layer]["correct"]) / len(iris[layer]["correct"])
        iris[layer]["miss"] = sum(iris[layer]["miss"]) / len(iris[layer]["miss"])
        iris[layer]["adversarial"] = sum(iris[layer]["adversarial"]) / len(iris[layer]["adversarial"])

    for layer, data in enumerate(digit):
        digit[layer]["correct"] = sum(digit[layer]["correct"]) / len(digit[layer]["correct"])
        digit[layer]["miss"] = sum(digit[layer]["miss"]) / len(digit[layer]["miss"])
        digit[layer]["adversarial"] = sum(digit[layer]["adversarial"]) / len(digit[layer]["adversarial"])

    for layer, data in enumerate(mnist):
        mnist[layer]["correct"] = sum(mnist[layer]["correct"]) / len(mnist[layer]["correct"])
        mnist[layer]["miss"] = sum(mnist[layer]["miss"]) / len(mnist[layer]["miss"])
        mnist[layer]["adversarial"] = sum(mnist[layer]["adversarial"]) / len(mnist[layer]["adversarial"])


    title = ["入力層", "第一中間層", "第二中間層", "第三中間層", "出力層"]
    blank = "－"
    for target in [["correct", "miss"], ["correct", "adversarial"]]:
        for i in range(len(title) - 1):
            print("{}&{}/{} &{}/{} &{}/{} &{}/{} \\\\ \cline".format(
                title[i],
                "{:.2f}".format(iris[i][target[0]]) if i < len(iris)-1 else blank,
                "{:.2f}".format(iris[i][target[1]]) if i < len(iris)-1 else blank,
                "{:.2f}".format(wine[i][target[0]]) if i < len(wine)-1 else blank,
                "{:.2f}".format(wine[i][target[1]]) if i < len(wine)-1 else blank,
                "{:.2f}".format(digit[i][target[0]]) if i < len(digit)-1 else blank,
                "{:.2f}".format(digit[i][target[1]]) if i < len(digit)-1 else blank,
                "{:.2f}".format(mnist[i][target[0]]) if i < len(mnist)-1 else blank,
                "{:.2f}".format(mnist[i][target[1]]) if i < len(mnist)-1 else blank,
            ) + r"{0-4}")
        for i in range(len(title)-1, len(title)):
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
        sum(iris_ad_acc)*100/len(iris_ad_acc),
        sum(wine_ad_acc)*100/len(wine_ad_acc),
        sum(digit_ad_acc)*100/len(digit_ad_acc),
        sum(mnist_ad_acc)*100/len(mnist_ad_acc),
    ))



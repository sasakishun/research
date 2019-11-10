def to_latex():
    return


if __name__ == '__main__':
    iris_correct = [['0.0000', '2.0000', '2.0000'],
                    ]
    iris_miss = [['0.1053', '0.0000', '0.0000'],
                    ]
    iris_adversarial = [['0.0000', '0.1767', '0.1067'],
                    ]
    wine_correct = [['0.1667', '0.0000', '0.0000', '0.0000', '0.0000'],
                    ]
    wine_miss = [['0.0000', '0.0000', '0.0000', '0.0000', '0.0000'],
                    ]
    wine_adversarial = [['0.0000', '0.8900', '0.3667', '0.4233', '0.2267'],
                    ]
    digit_correct = [['0.0848', '0.3000', '0.0242'],
                    ]
    digit_miss = [['0.8333', '2.9000', '1.8333'],
                    ]
    digit_adversarial = [['0.0000', '5.5390', '0.1820'],
                    ]
    mnist_correct = [['0.0089', '0.0113', '0.0001'],
                    ]
    mnist_miss = [['0.2654', '0.2165', '1.3911'],
                    ]
    mnist_adversarial = [['0.0000', '9.7130', '0.0000'],
                    ]

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
    for target in [["correct", "miss"], ["correct", "adversarial"]]:
        for i in range(len(title) - 1):
            print("{}&{}/{} &{}/{} &{}/{} &{}/{} \\\\ \cline".format(
                title[i],
                iris[i][target[0]] if i < len(iris)-1 else "－", iris[i][target[1]] if i < len(iris)-1 else "－",
                wine[i][target[0]] if i < len(wine)-1 else "－", wine[i][target[1]] if i < len(wine)-1 else "－",
                digit[i][target[0]] if i < len(digit)-1 else "－", digit[i][target[1]] if i < len(digit)-1 else "－",
                mnist[i][target[0]] if i < len(mnist)-1 else "－", mnist[i][target[1]] if i < len(mnist)-1 else "－",
            ) + r"{0-4}")
        for i in range(len(title)-1, len(title)):
            print("{}&{}/{} &{}/{} &{}/{} &{}/{} \\\\ \Hline".format(
                title[i],
                iris[-1][target[0]], iris[-1][target[1]],
                wine[-1][target[0]], wine[-1][target[1]],
                digit[-1][target[0]], digit[-1][target[1]],
                mnist[-1][target[0]], mnist[-1][target[1]],
            ) + r"{0-4}")
        print()



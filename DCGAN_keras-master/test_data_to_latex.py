def to_latex():
    return


if __name__ == '__main__':
    iris_miss = [
        ['0.0000', '2.0000', '2.0000'],
        ['0.0000', '2.0000', '2.0000'],
        ['0.0000', '1.0000', '2.0000'],
        ['1.0588', '3.5294', '2.2353'],
        ['0.0000', '1.0000', '2.0000'],
        ['0.0000', '3.0000', '2.0000'],
        ['0.0000', '1.0000', '2.0000'],
        ['0.0000', '2.0000', '2.0000'],
        ['0.0000', '1.0000', '2.0000'],
        ['0.0000', '2.0000', '2.0000'],
        ['0.0000', '2.0000', '2.0000']]

    iris_correct = [
        ['0.1053', '0.0526', '0.0000'],
        ['0.1053', '0.0263', '0.0000'],
        ['0.1053', '0.1053', '0.0000'],
        ['0.0904', '0.5190', '0.0321'],
        ['0.1053', '0.1316', '0.0000'],
        ['0.1053', '0.0789', '0.0000'],
        ['0.1053', '0.0263', '0.0000'],
        ['0.1053', '0.2105', '0.0000'],
        ['0.1053', '0.0263', '0.0000'],
        ['0.1053', '0.0789', '0.0000'],
        ['0.1053', '0.0526', '0.0000']]

    iris_adversarial = [
        ['1.5860', '0.5700', '0.1420'],
        ['1.6360', '0.9890', '0.0910'],
        ['1.5060', '1.4700', '0.1340'],
        ['1.7100', '1.6430', '0.1500'],
        ['1.5680', '1.5420', '0.1200'],
        ['1.5570', '0.8370', '0.2070'],
        ['1.5900', '1.5310', '0.1380'],
        ['1.5440', '1.2420', '0.0980'],
        ['1.5500', '1.1910', '0.1300'],
        ['1.6480', '1.0490', '0.0870']]

    iris_ad_acc = [0.5300, 0.6150, 0.6490, 0.6160, 0.5710, 0.5800, 0.6650, 0.6130, 0.6450, 0.6630]

    wine_miss = [
        ['0.0000', '0.0000', '1.0000', '1.0000', '2.0000'],
        ['1.0000', '0.0000', '1.0000', '1.0000', '0.0000'],
        ['1.0000', '0.0000', '1.0000', '1.0000', '0.0000'],
        ['0.0000', '1.0000', '0.0000', '0.0000', '2.0000'],
        ['1.0000', '0.0000', '0.0000', '0.0000', '2.0000'], ]

    wine_correct = [
        ['0.2778', '0.0000', '0.0000', '0.0000', '0.0000'],
        ['0.2778', '0.0000', '0.0000', '0.0000', '0.0000'],
        ['0.2941', '0.0000', '0.0000', '0.0000', '0.0000'],
        ['0.2778', '0.0000', '0.0000', '0.0000', '0.0000'],
        ['0.2353', '0.0000', '0.0000', '0.0000', '0.0000'],
        ['0.2222', '0.0556', '0.0556', '0.1111', '0.0000'],
        ['0.1765', '0.0588', '0.0000', '0.0000', '0.0000'],
        ['0.2222', '0.0000', '0.0556', '0.0000', '0.0556'],
        ['0.1765', '0.0588', '0.0000', '0.0000', '0.0000'],
        ['0.1667', '0.1111', '0.0556', '0.0556', '0.0000']]

    wine_adversarial = [
        ['4.2000', '0.3180', '0.2180', '0.2020', '0.1290'],
        ['4.1160', '0.2060', '0.0210', '0.0240', '0.0150'],
        ['4.2300', '0.3410', '0.1570', '0.0300', '0.0010'],
        ['3.8610', '0.6800', '0.5100', '0.2020', '0.1850'],
        ['3.9580', '0.5290', '0.2000', '0.1620', '0.0600'],
        ['3.1290', '0.6980', '0.7620', '0.6770', '0.0460'],
        ['3.3130', '1.9290', '0.8770', '0.4530', '0.2870'],
        ['3.3350', '1.0830', '0.6410', '0.5240', '0.0590'],
        ['3.3590', '0.6630', '0.2760', '0.2880', '0.1200'],
        ['4.0300', '1.9170', '0.4580', '0.4270', '0.1480']]

    wine_ad_acc = [0.9220, 0.8440, 0.8510, 0.8490, 0.8820, 0.7850, 0.8850, 0.8910, 0.7940, 0.8900]

    digit_miss = [
        ['1.1250', '3.8750', '2.1667'],
        ['1.3529', '5.2941', '2.1176'],
        ['1.0588', '3.5294', '2.2353'],
        ['1.2778', '4.0556', '2.1667'],
        ['0.8077', '4.3846', '2.1538'],
        ['1.0000', '4.7647', '2.0588'],
        ['1.0455', '4.2727', '2.0000'],
        ['1.1875', '3.6875', '2.2500'],
        ['1.1818', '3.8182', '2.0909'],
        ['1.1905', '4.3810', '2.0952']]

    digit_correct = [
        ['0.0655', '0.4911', '0.0446'],
        ['0.0758', '0.5423', '0.0321'],
        ['0.0904', '0.5190', '0.0321'],
        ['0.0789', '0.4591', '0.0205'],
        ['0.0838', '0.4521', '0.0269'],
        ['0.0933', '0.5364', '0.0204'],
        ['0.0769', '0.4408', '0.0414'],
        ['0.0872', '0.4826', '0.0378'],
        ['0.0769', '0.4911', '0.0148'],
        ['0.0708', '0.4897', '0.0649']]

    digit_adversarial = [
        ['12.8990', '15.2620', '0.3080'],
        ['13.1800', '18.9880', '0.4360'],
        ['14.1470', '17.7310', '0.4710'],
        ['11.8340', '15.4630', '0.1730'],
        ['15.0990', '18.0140', '0.5410'],
        ['11.4690', '12.7760', '0.2130'],
        ['12.3990', '16.5220', '0.4420'],
        ['12.8930', '17.2780', '0.3820'],
        ['13.7930', '13.6340', '0.6600'],
        ['13.8610', '15.6170', '0.4560']]

    digit_ad_acc = [1.0, 1.0, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, ]

    mnist_miss = [
        ['0.9032', '0.6036', '1.4590'],
        ['0.9042', '0.7743', '1.6955'],
        ['0.8813', '0.6347', '1.6651'],
        ['0.9177', '0.7229', '1.6580'],
        ['0.9860', '0.8118', '1.7294'],
        ['0.8184', '0.6260', '1.6450'],
        ['0.8974', '0.7379', '1.5897'],
        ['0.8491', '0.6681', '1.6178'],
        ['0.8785', '0.6738', '1.6754'],
        ['0.9170', '0.6987', '1.6317']]

    mnist_correct = [
        ['0.0786', '0.0534', '0.0013'],
        ['0.0649', '0.0965', '0.0026'],
        ['0.0653', '0.0999', '0.0012'],
        ['0.0647', '0.0900', '0.0021'],
        ['0.0617', '0.1016', '0.0021'],
        ['0.0624', '0.0872', '0.0014'],
        ['0.0655', '0.0909', '0.0034'],
        ['0.0643', '0.0911', '0.0018'],
        ['0.0666', '0.0872', '0.0021'],
        ['0.0610', '0.0999', '0.0017']]

    mnist_adversarial = [
        ['199.8540', '26.2640', '0.0000'],
        ['151.3360', '60.4290', '0.0060'],
        ['157.8790', '59.1640', '0.0200'],
        ['160.2090', '65.1110', '0.0020'],
        ['200.8480', '68.3830', '0.0000'],
        ['150.3160', '55.1360', '0.0050'],
        ['170.0880', '59.0390', '0.0110'],
        ['142.4910', '53.0910', '0.0000'],
        ['168.3090', '51.7010', '0.0130'],
        ['141.8830', '57.7480', '0.0040']]

    mnist_ad_acc = [1.0, 1.0, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]


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



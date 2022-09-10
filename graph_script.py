from cProfile import label
import matplotlib.pyplot as plt


def plot_w1_vs_w100():
    
    x = [epoch for epoch in range(30)]
    y_w1 = [
        0.6821, # 0
        0.8139, # 1
        0.8517, # 2
        0.8677, # 3
        0.8755, # 4
        0.8770, # 5
        0.8802, # 6
        0.8770, # 7
        0.8726, # 8
        0.8659, # 9
        0.8655, # 10
        0.8635, # 11
        0.8581, # 12
        0.8597, # 13
        0.8617, # 14
        0.8605, # 15
        0.8623, # 16
        0.8568, # 17
        0.8557, # 18
        0.8552, # 19
        0.8557, # 20
        0.8558, # 21
        0.8552, # 22
        0.8557, # 23
        0.8547, # 24
        0.8564, # 25
        0.8571, # 26
        0.8571, # 27
        0.8568, # 28
        0.8536, # 29
    ]

    y_w100 = [
        0.7541, # 0
        0.8364, # 1
        0.8779, # 2
        0.8937, # 3
        0.9001, # 4
        0.8976, # 5
        0.9035, # 6
        0.9016, # 7
        0.9001, # 8
        0.8981, # 9
        0.8939, # 10
        0.8933, # 11
        0.8933, # 12
        0.8883, # 13
        0.8901, # 14
        0.8871, # 15
        0.8879, # 16
        0.8902, # 17
        0.8867, # 18
        0.8882, # 19
        0.8888, # 20
        0.8843, # 21
        0.8866, # 22
        0.8862, # 23
        0.8820, # 24
        0.8876, # 25
        0.8879, # 26
        0.8879, # 27
        0.8877, # 28
        0.8841, # 29
    ]

    plt.plot(x, y_w1, x, y_w100)
    plt.show()




def plot_all_vs_all():
    """
    all W100
    """
    x = [epoch for epoch in range(30)]

    full = [
        0.7541, # 0
        0.8364, # 1
        0.8779, # 2
        0.8937, # 3
        0.9001, # 4
        0.8976, # 5
        0.9035, # 6
        0.9016, # 7
        0.9001, # 8
        0.8981, # 9
        0.8939, # 10
        0.8933, # 11
        0.8933, # 12
        0.8883, # 13
        0.8901, # 14
        0.8871, # 15
        0.8879, # 16
        0.8902, # 17
        0.8867, # 18
        0.8882, # 19
        0.8888, # 20
        0.8843, # 21
        0.8866, # 22
        0.8862, # 23
        0.8820, # 24
        0.8876, # 25
        0.8879, # 26
        0.8879, # 27
        0.8877, # 28
        0.8841, # 29
    ]

    disjoint = [
        0.7521, # 0
        0.8438, # 1
        0.8740, # 2
        0.8893, # 3
        0.8942, # 4
        0.9003, # 5
        0.8987, # 6
        0.8906, # 7
        0.8983, # 8
        0.8906, # 9
        0.8894, # 10
        0.8849, # 11
        0.8900, # 12
        0.8865, # 13
        0.8841, # 14
        0.8872, # 15
        0.8853, # 16
        0.8855, # 17
        0.8820, # 18
        0.8845, # 19
        0.8858, # 20
        0.8842, # 21
        0.8833, # 22
        0.8836, # 23
        0.8851, # 24
        0.8847, # 25
        0.8851, # 26
        0.8841, # 27
        0.8844, # 28
        0.8843, # 29
    ]


    # plt.plot(x, null, label="null")
    plt.plot(x, full, label="full")
    plt.plot(x, disjoint, label="disjoint")
    # plt.plot(x, overlap, label="overlap")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # plot_w1_vs_w100()
    plot_all_vs_all()
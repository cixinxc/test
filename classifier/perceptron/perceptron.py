from numpy import *
import matplotlib.pyplot as plt

def drawdata(train_x, train_y, train_label,bc,test_x):
    for i in range( train_x.shape[0] ):
        print(train_x[i, 0], train_x[i, 1])
        if train_label[0, i] == -1:
            plt.plot(train_x[i, 0], train_x[i, 1], 'or')
        elif train_label[0, i] == 1:
            plt.plot(train_x[i, 0], train_x[i, 1], 'ob')
    for i in range( test_x.shape[0] ):
        if train_label[0, 80+i] == -1:
            plt.plot(test_x[i, 0], test_x[i, 1], 'oy')
        elif train_label[0, 80+i] == 1:
            plt.plot(test_x[i, 0], test_x[i, 1], 'ow')

    train_y = train_y.T
    w, b = perceptron(train_x, train_y, bc)
    print(w, b)
    y1 = (-b - (2 * w[0, 0])) / (1.0 * w[0, 1])
    y2 = (-b - (6 * w[0, 0])) / (1.0 * w[0, 1])

    plt.plot([2,6], [y1, y2])
    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.title("a simple example")
    plt.plot([-6, -6], [4, 10], '-g')
    plt.show()

def perceptron(train_x, train_y, bc):
    w = [0.0, 0.0]
    w = mat(w)
    b = 0.0
    wc = 0.0
    count = 0
    error = 1
    print(train_x.shape)
    while True:
        if error == 0:
            break
        error = 0
        for i in range(train_x.shape[0]):
            wc = 0.0
            while True:
                wc = train_y[i, 0]*( train_x[i, 0] * w[0, 0] + train_x[i, 1] * w[0, 1] + b )
                if wc <= 0:
                    print(train_y[i, 0], train_x[i, 0],train_y[i, 0],w[0,0], w[0,1])
                    w[0, 0] += bc*train_y[i, 0]*train_x[i, 0]
                    w[0, 1] += bc*train_y[i, 0]*train_x[i, 1]
                    b += bc*train_y[i, 0]
                    count += 1
                    error += 1
                else:
                    break
                # if count == 100:
                #     break

    return w, b

def main():
    print('main function')
    dataSet = []
    labels = []
    fileIn = open('testSet1.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])
        labels.append(float(lineArr[2]))
    dataSet = mat(dataSet)
    labels = mat(labels)
    train_x = dataSet[0:81, :]
    train_y = labels[0:81, :]
    test_x = dataSet[80:101, :]
    test_y = labels[80:101, :]
    bc = 0.1
    drawdata(train_x, train_y, labels,bc,test_x)

main();
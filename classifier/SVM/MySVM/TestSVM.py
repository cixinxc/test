import random;
import numpy;
import copy;
import SVM;

def splitdata(all_x, all_y, radio):
    train_x = []
    train_y = []
    test_x = all_x
    test_y = all_y
    length = len(all_y)
    train_num = int(length * radio)
    for i in range(train_num):
        index = random.randrange(len(test_y))
        train_x.append(test_x.pop(index))
        train_y.append(test_y.pop(index))
    # 转换为矩阵
    train_x = numpy.array(train_x)
    train_y = numpy.array(train_y)
    test_x = numpy.array(test_x)
    test_y = numpy.array(test_y)
    print('数据拆分结果:{}训练数据矩阵{}    训练标签矩阵{}{}    测试数据矩阵{}    测试标签矩阵{}'.format('\n\t', train_x.shape, train_y.shape, '\n',test_x.shape, test_y.shape))
    return train_x, train_y, test_x, test_y

def splitdataPT(all_x, all_y, radio):
    train_x = []
    train_y = []
    test_x = all_x
    test_y = all_y
    length = len(all_y)
    train_num = int(length * radio)
    for i in range(train_num):
        train_x.append(test_x.pop(-1))
        train_y.append(test_y.pop(-1))
    # 转换为矩阵
    train_x = numpy.array(train_x)
    train_y = numpy.array(train_y)
    test_x = numpy.array(test_x)
    test_y = numpy.array(test_y)
    print('数据拆分结果:{}训练数据矩阵{}    训练标签矩阵{}{}    测试数据矩阵{}    测试标签矩阵{}'.format('\n\t', train_x.shape, train_y.shape, '\n',test_x.shape, test_y.shape))
    return train_x, train_y, test_x, test_y

def loadfile(filename):
    X = []
    Y = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            lineStr = line.strip().split('\t')
            length = len(lineStr)
            X.append( [float(lineStr[i]) for i in range(length - 1)] )
            Y.append(float(lineStr[-1]))
    return X, Y

def main():
    radio = 0.80
    filename = 'testSet2.txt'
    # 读取数据
    X, Y = loadfile(filename)
    # 拆分数据
    train_x, train_y, test_x, test_y = splitdata(copy.deepcopy(X), copy.deepcopy(Y), radio)
    # 训练SVM
    svmClassifier = SVM.train_SVM2(train_x, train_y, 0.6, 0.001,  kernelOption = ('rbf', 0))
    ## step 3: testing
    print("step 3: testing...")
    test_y = test_y.T
    accuracy = SVM.testSVM(svmClassifier, test_x, test_y)
    ## step 4: show the result
    print("step 4: show the result...")
    print('The classify accuracy is: %.3f%%' % (accuracy * 100))
    SVM.showSVM(svmClassifier)
    print(svmClassifier.kernelMat.shape)

main();
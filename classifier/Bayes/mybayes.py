import csv;
import math;
import random;
from timeit import Timer;

#从csv文件中读取数据，并转换为float
def load_cdv(filename):
    data = csv.reader(open(filename,'r'))
    dataset = list(data)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

#将dataset按 radio:(1-radio) 的比例分成训练数据和测试数据两部分
def split_dataset(dataset,radio):
    train_data = dataset
    test_data = []
    test_size = int(len(dataset)*(1-radio))
    while len(test_data)<test_size:
        test_data.append(train_data.pop(random.randrange(len(train_data))))
    return train_data, test_data

def split_dataset2(dataset ,radio):
    train_data = dataset
    test_data = []
    test_size = int(len(dataset) * (1 - radio))
    i=1
    while i < test_size:
        print(i)
        test_data.append(train_data.pop(i))
        i += 1
    return train_data, test_data

#求均值

def mean(numbers):
    return sum(numbers)/float( len(numbers) )

#求方差
def stdev(numbers):
    means = mean(numbers)
    # 无语了，忘记开方了，结果正确率一直在60左右
    return math.sqrt(sum([math.pow(x - means,2) for x in numbers])/float(len(numbers)-1))

#将训练数据中的数据按类别分开
def split_by_class(train_data):
    classed_data = {}
    for i in range(len(train_data)):
        if train_data[i][-1] not in classed_data:
            classed_data[train_data[i][-1]] = []
        classed_data[train_data[i][-1]].append(train_data[i])
    return classed_data

#计算数据中的每类每列的均值方差
def calculate_mean_and_stdev(classed_data):
    means_stdevs = {}
    for value, instance in classed_data.items():
        means_stdevs[value] = []
        for numbers in zip(*instance):
            means_stdevs[value].append([mean(numbers), stdev(numbers)] )
    '''
    取值方式
    for a,b in means_stdevs[0][1]:
        print(a,b)
    '''
    return means_stdevs

#计算一个数在高斯密度函数中的概率
def calculate_probability(number, mean, stdev):
    #无语了，下面这句写成了  a = math.exp(-(math.pow(number-mean,2)/2*math.pow(stdev,2)))  悲剧了
    a = math.exp(-(math.pow(number-mean,2)/(2*math.pow(stdev,2))))
    b = 1.0/float(math.sqrt(2*math.pi)*stdev)
    probabilities = a* b
    return probabilities


#计算一个向量对于各类的的联合概率
def calculate_class_probability(vector,means_stdevs):
    probabilities = {}
    for value, mean_stdev in means_stdevs.items():
        means, stdevs = zip(*mean_stdev)
        means = list(means)
        stdevs = list(stdevs)
        probabilities[value] = []
        p = 1.0
        for i in range(len(vector)):
            if i < 8:
                p *= calculate_probability(vector[i], means[i], stdevs[i])
        probabilities[value] = p
    return probabilities

def calculate_all_data(test_data, means_stdevs):
    result = {}
    for i in range(len(test_data)):
        result[i] = []
        result[i].append( calculate_class_probability(test_data[i], means_stdevs) )
    return result

def get_result(result,test_data):
    t_radio = 0
    for value, instance in result.items():
        type = None
        for v, ins in instance[0].items():
            if type == None  or tvalue<ins:
                type = v
                tvalue = ins
        if type == test_data[value][-1]:
            t_radio += 1
    print(t_radio/len(test_data)*100,'%')
    return t_radio


#主函数
def main1():
    filename = 'data.csv'
    radio = 0.67
    dataset =  load_cdv(filename)
    [train_data, test_data] = split_dataset(dataset, radio)
    classed_data = split_by_class(train_data)   #xx = { 0.0:[1] 1.1:[2] }中数据的取出: xx[0]或者xx[1.1]
    means_stdevs = calculate_mean_and_stdev(classed_data)
    test = [1,2,3]
    probabilities = calculate_class_probability(test, means_stdevs)
    result = calculate_all_data(test_data, means_stdevs)
    get_result(result, test_data)
    return
def main():
    t1=Timer("main1()","from __main__ import main1")
    print(t1.timeit(10))
    return

main();
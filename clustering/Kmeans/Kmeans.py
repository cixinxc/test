import csv;
import math;
import copy;
import random;

def load_csv(filename):
    data = csv.reader(open(filename,'r'))
    dataset = list(data)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def calculate_distance(v1, v2):
    distance = 0
    distance = math.sqrt(sum(math.pow( v2[i]-v1[i] ,2) for i in range(len(v2))))
    return distance

#{}中添加[]，[]中怎么添加[]
def get_K_center(dataset, K, radio):
    some_data = []
    center = []
    total = len(dataset)
    some_size = int(total*radio)
    while len(some_data) < some_size:
        some_data.append(dataset[random.randrange(total)])
    means = []
    for numbers in zip(*some_data):
        means.append(mean(numbers))
    del means[-1]

    print('随机抽取数据的均值:',[ x for x in means])
    center.append([ x*0.5 for x in means])
    center.append([ x*1.5 for x in means])
    # 注意，随机初始化聚类中心，正确率无法保证 33/66
    # cs = []
    # for i in range(8):
    #     cs.append(random.randrange(100))
    # center.append(cs)
    #
    # cs2 = []
    # for i in range(8):
    #     cs2.append(random.randrange(100))
    # center.append(cs2)
    #
    # print('初始聚类中心')
    # print(cs)
    # print(cs2)
    # print('')

    print('center  ',center)
    return center

def re_center(data):
    centers = []
    c1 = []
    c2 = []
    means = []
    means2 = []
    for i in range(len(data)):
        if data[i][-1] == 0.0:
            c1.append(data[i])
        else:
            c2.append(data[i])
    center = []
    for numbers in zip(*c1):
        means.append(mean(numbers))
    del means[-1]
    center.append([x  for x in means])

    for numbers in zip(*c2):
        means2.append(mean(numbers))
    del means2[-1]
    center.append([x for x in means2])

    print('centers',center)
    return center

def Kmeans(dataset, K):
    radio = 0.33
    k_max = 1000
    change = 0
    centers = get_K_center(dataset, K, radio)
    data = copy.deepcopy(dataset)
    i = 0
    while i < k_max:
        for index in range(len(data)):
            s_class = data[index][-1]
            if calculate_distance(data[index], centers[0]) < calculate_distance(data[index], centers[1]):
                data[index][-1] = 0.0
            else:
                data[index][-1] = 1.0
            if s_class != data[index][-1]:
                change += 1
        print('第',i+1,'次聚类完成，','重分数据个数:',change)
        centers = re_center(data)
        print('centers', len(centers))
        if change == 0:
            break
        i += 1
        change = 0
    return data

def acc(data, dataset):
    count = 0
    for i in range(len(data)):
        if data[i][-1] == dataset[i][-1]:
            count += 1
    print(count/len(data)*100,'%')
    return 0

def main():
    filename = 'data.csv'
    radio = 0.33
    K = 2
    dataset = load_csv(filename)
    data = Kmeans(dataset, K)
    acc(data, dataset)
    return 0
main();
import csv;
import copy;
import math;
import random;

def load_csv(filename):
    data = csv.reader(open(filename,'r'))
    dataset = list(data)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def split_data(dataset, radio):
    train_data = dataset
    test_data = []
    test_size = int(len(train_data)*(1-radio))
    while len(test_data) < test_size:
        test_data.append(train_data.pop(random.randrange(len(train_data))))
    return train_data, test_data

#欧氏距离:Euclidean
def calculate_distance(tdata, data, type):
    distance = 0.0
    if type == 'Euclidean':
        sum = 0
        for i in range(len(tdata)):
            sum += math.pow(tdata[i] - data[i],2)
        distance = math.sqrt(sum)
    elif type == '':
        distance = 0.0
    else:
        distance = 0.0
    return distance

def calculate_all_data(test_data, train_data, K):
    result = {}
    for i in range(len(test_data)):
        #print('第',i,'个数据',test_data[i])
        result[i] = calculate_class(test_data[i], train_data, K)
    return result

def calculate_class(tdata, train_data, K):
    result = 0
    closest = {}
    train_datas = train_data
    del tdata[-1]
    for i in range(len(train_data)):
        #到底是引用还是指针？
        #del train_datas[i][-1]
        closest = sequence_insert(i, calculate_distance(tdata, train_datas[i], 'Euclidean'), closest, K)
    #print(closest)
    return closest

def sequence_insert(index, ins, lists, K):
    insert_p = 0
    if len(lists) == 0:
        lists[0] = [index, ins]
        #print('ss',lists[0][0],lists[0][1])
    else:
        if len(lists) < K:
            length = len(lists)
        else:
            length = K

        for i in range(length)[::-1]:
            if ins > lists[i][1]:
                break;
            if i == 0:
                i -= 1
        i += 1

        end = length
        while i < end:
            lists[end] = lists[end-1]
            end -= 1
        lists[end] = [index, ins]
    return lists

def tongji(result, test_data, train_data, K):
    count = 0
    for i in range(len(result)):
        fenlei = 0
        for j in range(len(result[i])):
            #print(train_data[result[i][j][0]][-1])
            if train_data[result[i][j][0]][-1] == 0.0:
                fenlei += 1
        type = None
        if fenlei < K/2 :
            type = 0.0
        else:
            type = 1.0

        #print('test_data[i][-1',test_data[i][-1],'    ',type)
        if test_data[i][-1] == type:
            count += 1
    print(count*1.0/len(test_data)*100,'%')
    return count

def main():
    file = 'data.csv'
    radio = 0.67
    K = 5
    dataset = load_csv(file)
    train_data, test_data = split_data(dataset,radio)
    test_datas = copy.deepcopy(test_data)
    test1 = [0, 0, 0]
    test2 = [0, 3, 4,7]
    a = calculate_distance(test1, test2, 'Euclidean')
    result = calculate_all_data(test_datas,train_data, K)
    tongji(result, test_data, train_data, K)
    return 0

main();
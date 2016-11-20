import csv
import random
import math

def load_csv(file_name):
    data = csv.reader(open(file_name,'r'))
    dataset = list(data)
    #此时print(list(data)) 会显示:[]
    for i in range( len(dataset) ):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

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

def splitDataset2(dataset ,radio):
    d1 = dataset
    d2=[]
    size = int( len(dataset)*radio )
    while len(d2)<size :
        i = random.randrange(len(dataset))
        d2.append(d1.pop(i))
    return [d1,d2]

def splitDataset(dataset,radio):
    traindata = dataset
    testsize = int(len(dataset)*(1-radio))
    testdata = []
    i = 0
    while len(testdata) < testsize:
        testdata.append(traindata.pop(i))
        i +=1
    return traindata, testdata

def separate_by_class(dataset):
    sperates = {}
    for x in range(len(dataset)):
        if dataset[x][-1] not in sperates:
            sperates[dataset[x][-1]] = []
        sperates[dataset[x][-1]].append(dataset[x])
    return sperates

def SeparateByClass(dataset):
    sperates = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in sperates:
            sperates[ vector[-1] ] = []
        sperates[vector[-1]].append(vector)
    return sperates

def mean(numbers):
    return sum(numbers)/float( len(numbers) )



def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/  float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for value,instances in separated.items():
        summaries[value] = summarize(instances)
    return summaries

def summarizeByClass(dataset):
    separated = SeparateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            probabilities[classValue] *= calculateProbability( inputVector[i], mean,stdev )
    return probabilities

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculate_class_probabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:

            bestProb = probability
            bestLabel = classValue
        print(classValue, probability)
    print('')
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def main():
    filename = 'data.csv'
    splitRatio = 0.67
    dataset = load_csv(filename)
    #    load_csv_data(filename)
    trainingSet, testSet = splitDataset2(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    # prepare model
    summaries = summarize_by_class(trainingSet)
    print('assss   ',summaries)
    # test model
    predictions = getPredictions(summaries, testSet)

    accuracy = getAccuracy(testSet, predictions)
    print("Accuracy: {0}%".format(accuracy))
main();

# coding: utf-8

"""

Реализовать метрический классификатор kNN
Сделать кросс-валидацию; обосновать выбор числа фолдов для нее
Выполнить визуализацию данных
Настроить классификатор с 2-3 метриками и 2-3 пространственными преобразованиями
Для оценки качества можно использовать метрику accuracy, но лучше - f1-measure

"""


# In[235]:


import math
import random
import numpy as np
import itertools
from matplotlib.colors import ListedColormap
import operator
import copy


# In[236]:


smallNumber = 0.000000000000000001


# In[237]:


class DataPoint:
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
        
    def __str__(self):
        return "(%f,%f) = {%d}" % (self.x, self.y, self.label)


# In[238]:


def showDataPoints(data):
    print(np.array([str(i) for i in data]).T)


# In[239]:


def calculateMinkowski(point1, point2, p=2):
    return pow(abs(point1.x-point2.x)**p + abs(point1.y-point2.y)**p, 1/p)


# In[240]:


def augmentData(data, params):
    result = copy.deepcopy(data)
    
    if "rotate" in params:
        result = rotateRandom(data)
    if "vert_flip" in params:
        result = flipVertical(data)
    if "horiz_flip" in params:
        result = flipHorizontal(data)
    if "noise" in params:
        result = noiseRandom(data)
    return result

def rotateRandom(data):
    randomDegree = random.randint(-180, 180)
    result = copy.deepcopy(data)
    for item in result:
        rotatePoint(item, randomDegree)
    return result

def rotatePoint(point, degree):
    x = point.x
    y = point.y
    rad = math.radians(degree)
    x1 = x * math.cos(rad) - y * math.sin(rad)
    y1 = x * math.sin(rad) + y * math.cos(rad)
    point.x = x1
    point.y = y1

def noiseRandom(data, mu = 0, sigma = 0.09):
    result = copy.deepcopy(data)
    noise = np.random.normal(mu, sigma, (len(data),2)) 
    for i in range(len(data)):
        result[i].x += noise[i][0]
        result[i].y += noise[i][1]
    return result

def flipHorizontal(data):
    result = copy.deepcopy(data)
    for item in result:
        item.x *=(-1)
    return result

def flipVertical(data):
    result = copy.deepcopy(data)
    for item in result:
        item.y *=(-1)
    return result


# In[241]:


##Reads data from file
def readData(path):
    file = open(path)
    data = []
    for line in file:
        splittedLine = line.replace(",", ".").split("\t")
        x = float(splittedLine[0])
        y = float(splittedLine[1])
        label = int(splittedLine[2])
        data.append(DataPoint(x,y,label))
    file.close()
    return data


# In[242]:


##Draws dots according to class
def showData(data):
    classColormap = ListedColormap(['#FF0000', '#00FF00'])
    pl.scatter([data[i].x for i in range(len(data))],
               [data[i].y for i in range(len(data))],
               c=[data[i].label for i in range(len(data))],
               cmap=classColormap)
    pl.show()


# In[243]:


##Separates data
def splitData(data, testSize):
    random.shuffle(data)
    learningCount = len(data) -  int(len(data) * testSize)
    return data[0:learningCount], data[learningCount+1:]


# In[244]:


def predict(trainData, testData, labels, kNeighbors, power = 2):
    predictions = []
    for testPoint in testData:
        distances = []
        # Calculate all distances between test point and other points
        for trainPoint in trainData:
            dist = calculateMinkowski(testPoint, trainPoint, power)
            distances.append((trainPoint.label, dist))
        distances.sort(key=operator.itemgetter(1)) 
       
        neighbors = [[i, 0] for i in range(labels)]
        # How many points of each class are near
        for item in distances[:kNeighbors]:
            neighbors[item[0]][1] += 1
            # Assign a class with the most number of occurences among K nearest neighbours
        predictedLabel = max(neighbors, key=operator.itemgetter(1))[0]
        predictions.append(predictedLabel)
    return predictions


# In[245]:


""" RETURN (recall, specificity, precision, accuracy, fscore)"""
def calculateParameters(testData, predictions):
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0

    for i in range(len(predictions)):
        #print(str(testData[i].label) +"    " + str(predictions[i]))
        if (testData[i].label == 0) and (predictions[i] == 0):
            trueNegative += 1
        if (testData[i].label == 0) and (predictions[i] == 1):
            falsePositive += 1
        if (testData[i].label == 1) and (predictions[i] == 1):
            truePositive += 1
        if (testData[i].label == 1) and (predictions[i] == 0):
            falseNegative += 1
  
    positive = smallNumber if (truePositive + falseNegative) == 0 else truePositive + falseNegative
    negative = smallNumber if (trueNegative + falsePositive) == 0 else trueNegative + falsePositive
    recall = truePositive / positive
    specificity = trueNegative / negative
    precision = truePositive / (smallNumber if (truePositive + falsePositive) == 0 else truePositive + falsePositive)
    accuracy = (truePositive + trueNegative) / (positive + negative)
    fscore=2*(precision*recall)/(smallNumber if (precision+recall) == 0 else precision+recall)
    
    return (recall, specificity, precision, accuracy, fscore)


# In[246]:


def showDifference(testData, predictions):
    classColormap = ListedColormap(['#FF0000', '#00FF00','#F08080','#90EE90'])
    colors=[]
    for i in range(len(predictions)):
        if (predictions[i]==0 and testData[i].label==0):
            colors.append([0,50])
        if (predictions[i]==1 and testData[i].label==1):
            colors.append([1,50])
        if (predictions[i]==1 and testData[i].label==0):
            colors.append([2,50])
        #when it's been predicted as red but it is actually green
        if (predictions[i]==0 and testData[i].label==1):
            colors.append([3,50])

    pl.scatter([testData[i].x for i in range(len(testData))],
               [testData[i].y for i in range(len(testData))],
               c=[colors[i][0] for i in range(len(colors))],
               s=[colors[i][1] for i in range(len(colors))],
               cmap=classColormap)
    pl.show()


# In[247]:


def generateData(data):
    #showDataPoints(data)
    return copy.deepcopy(data) + augmentData(data, "rotate, horiz_flip") + augmentData(data, "horiz_flip, vert_flip") 


# In[248]:


def getCrossValidation(data, numFolds):
    foldLength = round(len(data) / numFolds)
    
    trainDatas = []
    testDatas = []
   
    for i in range(numFolds):
        n = (numFolds-i) * foldLength 
        trainData = data[: n - foldLength] + data[n:]
        testData = data[n - foldLength : n]
        
        trainDatas.append(trainData)
        testDatas.append(testData)
    return (trainDatas, testDatas)  


# # Main

# ## Constants

# In[326]:


testSize = 0.2
numberOfLabels = 2
numberOfNeighbors = 5
numberOfFolds = 4
fileName = 'chips.txt'


# In[327]:


data = readData(fileName)


# ### Ordinary sequence

# In[328]:


augmentedData = generateData(data)
showData(augmentedData)
trainData, testData = splitData(augmentedData, testSize)
predictions = predict(trainData, testData, numberOfLabels, numberOfNeighbors)
params = calculateParameters(testData, predictions)

showData(trainData)
showDifference(testData, predictions)

print("Recall: " + str(params[0]))
print("Specificity: " + str(params[1]))
print("Precision: " + str(params[2]))
print("Accuracy: " + str(params[3]))
print("F-score: " + str(params[4]))


# ### CrossValidation Average Parameters

# In[329]:


def calculateAverage(paramsArray, index):
    s = 0
    for item in paramsArray:
        s += item[index]
    return s / len(paramsArray)


# In[330]:


augmentedData = generateData(data)
trainDatas, testDatas = getCrossValidation(augmentedData, numberOfFolds)

crossParams = []

for i in range(len(trainDatas)):
    predictions = predict(trainDatas[i], testDatas[i], numberOfLabels, numberOfNeighbors)
    params = calculateParameters(testDatas[i], predictions)
    crossParams.append(params)

print("Average Recall: " + str(calculateAverage(crossParams, 0)))
print("Average Specificity: " + str(calculateAverage(crossParams, 1)))
print("Average Precision: " + str(calculateAverage(crossParams, 2)))
print("Average Accuracy: " + str(calculateAverage(crossParams, 3)))
print("Average F-score: " + str(calculateAverage(crossParams, 4)))


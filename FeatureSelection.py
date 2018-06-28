import numpy as np
import perceptron as pt
import sklearn.svm as svm



def zerosInARow(vector, k):
    if len(vector) < k:
        return 0
    vector = vector[-k:]
    zeros = []
    counter = 0
    for i in vector:
        if i == 0:
            counter += 1
        else:
            zeros.append(counter)
            counter = 0
    return np.mean(np.asarray(zeros), dtype=np.int)


def onesInARow(vector, k):
    if len(vector) < k:
        return 0
    vector = vector[-k:]
    ones = []
    counter = 0
    for i in vector:
        if i == 1:
            counter += 1
        else:
            ones.append(counter)
            counter = 0
    return np.mean(np.asarray(ones), dtype=np.int)


def numOfAlternations(vector, k):
    if len(vector) < k:
        return 0
    vector = vector[-k:]
    counter = 0
    for i in range(k - 1):
        if vector[i] != vector[i + 1]:
            counter += 1
    return counter


def numberOfOnes(vector, k):
    if len(vector) < k:
        return 0
    return np.histogram(vector[-k:],2)


def numOfIncreasingSeqs(vector, k):
    """
    :param vector:
    :param k:
    :return: number of increasing sequences of ones
    """
    if len(vector) < k:
        return 0
    vector = vector[-k:]
    ones = []
    counter = 0
    for i in vector:
        if i == 1:
            counter += 1
        else:
            ones.append(counter)
            counter = 0
    counterForIncreasingSeq = 0
    for i in range(len(ones) - 1):
        if ones[i + 1] > ones[i]:
            counterForIncreasingSeq += 1
    return counterForIncreasingSeq


def getFeatures(vector, k):
    features = []
    features.append(zerosInARow(vector, k))
    features.append(onesInARow(vector, k))
    # features.append(numOfAlternations(vector, k))
    # features.append(numberOfOnes(vector, k)[0][1])
    # features.append(numOfIncreasingSeqs(vector, k))
    return features


file = open("human.txt", "r")
vectors = []
line = file.readline()
while line:
    vectors.append(line)
    line = file.readline()

finalVectors = []
for line in vectors:
    vector = []
    for letter in line:
        if letter != "\n":
            vector.append(int(letter))
    finalVectors.append(vector)

# finalVectors = np.asarray(finalVectors)
# forTrain = np.random.random_integers(0, len(finalVectors), len(finalVectors) * 0.9)

forTrain = finalVectors[:1000]
forTest = finalVectors[1000:]

trainX = []
trainY = []
testX = []
testY = []

for vector in forTrain:
    trainX.append(vector[-50: -1])
    trainY.append(vector[-1:][0])

for vector in forTest:
    testX.append(vector[-50: -1])
    testY.append(vector[-1:][0])

X_train = []
X_test = []
for train in trainX:
    X_train.append(getFeatures(train, 40))
for test in testX:
    X_test.append(getFeatures(test, 40))

# ptr = pt.Perceptron()
X_train = np.asarray(X_train)
trainY = np.asarray(trainY)
svc = svm.SVC(C=1e10, kernel='linear')
print(X_train)
svc.fit(X_train, trainY)




print(getFeatures([0,0,0,1,1,1,0,1,1,1,0,1,1],10))
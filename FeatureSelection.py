import numpy as np

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
    return np.mean(np.asarray(zeros))


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
    return np.mean(np.asarray(ones))


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
    features.append(numOfAlternations(vector, k))
    features.append(numberOfOnes(vector, k)[0][1])
    features.append(numOfIncreasingSeqs(vector, k))
    return features



print(getFeatures([0,0,0,1,1,1,0,1,1,1,0,1,1],10))
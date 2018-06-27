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


def meanOfIncreasingSeqs(vector, k):
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
    numberOfSeqs = []
    for i in range(len(ones) - 1):
        if ones[i] > ones[i + 1]:
            counterForIncreasingSeq += 1
        elif ones[i] < ones[i + 1]:
            numberOfSeqs.append(counterForIncreasingSeq)
            counterForIncreasingSeq = 0
    return np.mean(np.asarray(counterForIncreasingSeq))




print(meanOfIncreasingSeqs([0,0,0,1,1,1,0,1,1,1,0,1,1],10))
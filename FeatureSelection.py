import numpy as np

def zerosInARow(vector, k):
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
    vector = vector[-k:]
    counter = 0
    for i in range(k - 1):
        if vector[i] != vector[i + 1]:
            counter += 1
    return counter


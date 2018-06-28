import numpy as np
import matplotlib.pyplot as plt
import editdistance as ed
from matplotlib.pyplot import *
# from keras.layers import SimpleRNN, TimeDistributed, Dense
# from keras.models import Sequential

d = np.loadtxt('human.txt', dtype=str)
A = []
# m = {'0': 0, '1': 1}
for x in d:
    if len(x) > 70:
        # z = np.array([int(c) for c in x])
        A.append(x)
A = np.array(A)


def draw(data, test_size, xl, yl):
    np.random.shuffle(data)
    # train_data = data[test_size:, :-20]
    # train_labels = data[test_size:, -20:]
    # test_data = data[:test_size, :-20]
    # test_labels = data[:test_size, -20:]

    train_data = data[test_size:]
    train_labels = data[test_size:]
    train_data = np.array([list(map(int, s[-xl - yl:-yl])) for s in train_data])
    train_labels = np.array([list(map(int, s[-yl:])) for s in train_labels])

    test_data = data[:test_size]
    test_labels = data[:test_size]
    test_data = np.array([list(map(int, s[-xl - yl:-yl])) for s in test_data])
    test_labels = np.array([list(map(int, s[-yl:])) for s in test_labels])

    return (train_data, train_labels), (test_data, test_labels)
    # return (np.expand_dims(train_data, axis=2), np.expand_dims(train_labels, axis=2)), \
    #        (np.expand_dims(test_data, axis=2), np.expand_dims(test_labels, axis=2))


train_data, test_data = draw(A, test_size=100, xl=50, yl=20)
train_data, train_labels = train_data
test_data, test_labels = test_data
def PCA():
    covM = np.cov(train_data)
    w, v = np.linalg.eig(covM)
    proj = [[] for _ in range(len(test_data))]
    for i in range(15):
        proj
    scatter(proj[0, :], proj[1, :])
    show()

PCA()
# p = learner.knn(30)
# p.fit(train_data, train_labels)

#
# def RNN(x, y):
#     model = Sequential()
#     for _ in range(10):
#         model.add(SimpleRNN(input_dim=1, output_dim=20, return_sequences=True))
#         model.add(TimeDistributed(Dense(output_dim=1, activation="sigmoid")))
#     model.compile(loss="hinge", optimizer="rmsprop")
#     model.fit(x, y, nb_epoch=10, batch_size=32)
#
#     return model
#
#
# model = RNN(train_data, train_labels)
# l = 0
# for x, y in zip(test_data, test_labels):
#     res = model.predict(np.expand_dims(x, 2)).ravel()
#     thr = np.mean(res)
#     res[res >= thr] = 1
#     res[res != 1] = 0
#     l += (res != y.ravel()).sum()
# print(l/ len(test_data))
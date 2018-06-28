import numpy as np
import learner
import matplotlib.pyplot as plt
import editdistance as ed

from keras.layers import SimpleRNN, TimeDistributed, Dense
from keras.models import Sequential


def prepare_data(data):
    pass


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

    # return (train_data, train_labels), (test_data, test_labels)
    return (np.expand_dims(train_data, axis=2), np.expand_dims(train_labels, axis=2)), \
           (np.expand_dims(test_data, axis=2), np.expand_dims(test_labels, axis=2))

# def knn_procedure(k, test_size,l):
#     train_data, test_data = draw(A, test_size,)
#     train_data, train_labels = train_data
#     test_data, test_labels = test_data
#     p = learner.knn(k)
#     p.fit(train_data, train_labels)
#     return p.score(test_data, test_labels)

K = range(1, 50, 2)
repeat = 10
knn_res = []
test_size = 100

# for k in K:
#     res = sum(knn_procedure(45, test_size,k) for _ in range(repeat)) / repeat
#     knn_res.append(res)

# print(np.mean(list(map(str.__len__,A))))
# plt.hist(list(map(str.__len__,A)),80)
# plt.show()

# plt.figure(0)
# plt.title("K nearest accuracy ratio")
# plt.xticks(K)
# plt.xlabel('m')
# plt.ylabel('accuracy ratio')
#
# plt.plot(K, knn_res)
# plt.show()

train_data, test_data = draw(A, test_size=100, xl=50, yl=20)
train_data, train_labels = train_data
test_data, test_labels = test_data

# p = learner.knn(30)
# p.fit(train_data, train_labels)


def RNN(x, y):
    mod = Sequential()
    mod.add(SimpleRNN(input_dim=50, output_dim=20, return_sequences=True))
    # mod.add(TimeDistributed(Dense(output_dim=1, activation="sigmoid")))
    mod.compile(loss="mse", optimizer="Adam")
    mod.fit(x, y, batch_size=20)
    # print(mod.predict(np.array([[[1], [0], [0], [0], [0], [0]]])))


RNN(train_data, train_labels)

import numpy as np
import learner
import matplotlib.pyplot as plt
import editdistance as ed


def prepare_data(data):
    pass


d = np.loadtxt('human.txt', dtype=str)
A = []
B = []
# m = {'0': 0, '1': 1}
for x in d:
    if len(x) > 70:
        # z = np.array([int(c) for c in x])
        A.append(x)
A = np.array(A)


def draw(data, test_size,l):
    np.random.shuffle(data)
    # train_data = data[test_size:, :-20]
    # train_labels = data[test_size:, -20:]
    # test_data = data[:test_size, :-20]
    # test_labels = data[:test_size, -20:]
    train_data = data[test_size:]
    train_labels = data[test_size:]
    train_data = [s[-l:-1] for s in train_data]
    train_labels = [int(s[-1]) for s in train_labels]

    test_data = data[:test_size]
    test_labels = data[:test_size]
    test_data = [s[-l:-1] for s in test_data]
    r = [int(s[-1]) for s in test_labels]
    r = np.array(r, dtype=int)

    return (np.array(train_data), np.array(train_labels)), (
        np.array(test_data), r)


def knn_procedure(k, test_size,l):
    train_data, test_data = draw(A, test_size,l)
    train_data, train_labels = train_data
    test_data, test_labels = test_data
    p = learner.knn(k)
    p.fit(train_data, train_labels)
    return p.score(test_data, test_labels)


K = range(1,50,2)
repeat = 10
knn_res = []
test_size = 100

for k in K:
    res = sum(knn_procedure(45, test_size,k) for _ in range(repeat)) / repeat
    knn_res.append(res)

# print(np.mean(list(map(str.__len__,A))))
# plt.hist(list(map(str.__len__,A)),80)
# plt.show()

plt.figure(0)
plt.title("K nearest accuracy ratio")
plt.xticks(K)
plt.xlabel('m')
plt.ylabel('accuracy ratio')

plt.plot(K, knn_res)
plt.show()

# train_data, test_data = draw(A, 100)
# train_data, train_labels = train_data
# test_data, test_labels = test_data
# p = learner.knn(30)
# p.fit(train_data, train_labels)


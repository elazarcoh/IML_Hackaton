import numpy as np
import editdistance
import matplotlib.pyplot as plt

d = np.loadtxt('human.txt', dtype=str)
A = []
for x in d:
    if len(x) > 70:
        A.append(x)
A = np.array(A)


class knn:
    def __init__(self, k, num_bits):
        self.fitted = False
        self.k = k
        self.num_bits = num_bits

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.fitted = True

    def predict(self, x):
        pred = x
        for _ in range(self.num_bits):
            pred += str(self.predict_bit(pred))
        return pred[-self.num_bits:]

    def predict_bit(self, x):
        assert self.fitted
        t = self.y[self.k_nearest_neighbors_indices(x)]
        z = [int(s[-1]) for s in t]
        k_near = int(np.round(np.mean(z)))
        return k_near

    def k_nearest_neighbors_indices(self, x):
        dist = np.array([editdistance.eval(x[-20:], y[-20:]) for y in self.X])
        k_nearest = np.argsort(dist)[:self.k]
        return k_nearest

    def score(self, data, true_labels):
        predictions = np.array([self.predict(x) for x in data])
        return np.mean((true_labels == predictions).sum(axis=0) / len(true_labels))


def draw(data, test_size, xl, yl):
    np.random.shuffle(data)
    train_data = data[test_size:]
    train_data = np.array([s[-xl - yl:-yl] for s in train_data])

    train_labels = data[test_size:]
    train_labels = np.array([s[-yl:] for s in train_labels])

    test_data = data[:test_size]
    test_data = np.array([s[-xl - yl:-yl] for s in test_data])

    test_labels = data[:test_size]
    test_labels = np.array([s[-yl:] for s in test_labels])

    return (train_data, train_labels), (test_data, test_labels)


def knn_procedure(k, test_size):
    train_data, test_data = draw(A, test_size, 50, 20)
    train_data, train_labels = train_data
    test_data, test_labels = test_data
    p = knn(k, 20)
    p.fit(train_data, train_labels)
    return p.score(test_data, test_labels), p


K = [2, 6, 12, 18]
repeat = 3
knn_res = []
test_size = 100
ps = []
for k in K:
    res = 0
    for _ in range(repeat):
        t, p = knn_procedure(k, test_size)
        res += t
    ps.append(p)
    knn_res.append(res / repeat)

plt.figure(0)
plt.title("K nearest accuracy ratio")
plt.xticks(K)
plt.xlabel('m')
plt.ylabel('accuracy ratio')

plt.plot(K, knn_res)
plt.show()

train_data, test_data = draw(A, test_size, 50, 20)
train_data, train_labels = train_data
test_data, test_labels = test_data
p = ps[3]
z1 = p.predict(test_data[0])
z2 = test_labels[0]
print(editdistance.eval(z1, z2))
print(z1)
print(z2)

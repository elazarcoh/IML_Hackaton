import numpy as np
import editdistance
import matplotlib.pyplot as plt

d = np.loadtxt('human.txt', dtype=str)
A = []
for x in d:
    if len(x) > 70:
        A.append(x)
A = np.array(A)


def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


class knn:
    def __init__(self, k, num_bits_to_guess, input_num_bits):
        self.fitted = False
        self.k = k
        self.num_bits = num_bits_to_guess
        self.input_num_bits = input_num_bits

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
        z = [int(s[0]) for s in t]
        k_near = int(np.round(np.mean(z)))
        return k_near

    def k_nearest_neighbors_indices(self, x):
        # dist = np.array([editdistance.eval(x[-20:], y[-20:]) for y in self.X])
        dist = np.array(
            [hamming2(x[-self.input_num_bits:], y[-self.input_num_bits:]) for y in self.X])
        k_nearest = np.argsort(dist)[:self.k]
        return k_nearest

    def score(self, data, true_labels):
        predictions = np.array([self.predict(x) for x in data])
        g = (1 - (hamming2(x, y) / len(x)) for x, y in zip(predictions, true_labels))
        return sum(g) / len(true_labels)
        # return np.mean((true_labels == predictions).sum(axis=0) / len(true_labels))


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
    train_data, test_data = draw(A, test_size, 30, 20)
    train_data, train_labels = train_data
    test_data, test_labels = test_data
    p = knn(k, 20, 30)
    p.fit(train_data, train_labels)
    return p.score(test_data, test_labels), p


K = [7]
repeat = 5
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
# plt.show()

train_data, test_data = draw(A, test_size, 50, 20)
train_data, train_labels = train_data
test_data, test_labels = test_data
for p in ps:
    mm = 0
    for x, y in zip(test_data, test_labels):
        z1 = p.predict(x)
        z2 = y
        mm += hamming2(z1, z2)
    print(mm / len(test_labels))

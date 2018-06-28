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
    def __init__(self, k):
        self.fitted = False
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.fitted = True

    def predict(self, x):
        assert self.fitted
        k_near = int(np.round(np.mean(self.y[self.k_nearest_neighbors_indices(x)])))

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
    train_labels = np.array([s[-xl - yl:-yl] for s in train_labels])

    test_labels = data[:test_size]
    train_labels = np.array([s[-yl:] for s in train_labels])

    return (train_data, train_labels), (test_data, test_labels)


def knn_procedure(k, test_size, l):
    train_data, test_data = draw(A, test_size, 50, 1)
    train_data, train_labels = train_data
    test_data, test_labels = test_data
    p = knn(k)
    p.fit(train_data, train_labels)
    return p.score(test_data, test_labels)


K = range(1, 50, 2)
repeat = 10
knn_res = []
test_size = 100

for k in K:
    res = sum(knn_procedure(45, test_size, k) for _ in range(repeat)) / repeat
    knn_res.append(res)

print(np.mean(list(map(str.__len__, A))))
plt.hist(list(map(str.__len__, A)), 80)
plt.show()

plt.figure(0)
plt.title("K nearest accuracy ratio")
plt.xticks(K)
plt.xlabel('m')
plt.ylabel('accuracy ratio')

plt.plot(K, knn_res)
plt.show()

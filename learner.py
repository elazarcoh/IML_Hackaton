import numpy as np
import editdistance


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
        # dist = np.linalg.norm(self.X - x, axis=1)
        dist = np.array([editdistance.eval(x[-20:],y[-20:]) for y in self.X])
        k_nearest = np.argsort(dist)[:self.k]
        return k_nearest

    def score(self, data, true_labels):
        predictions = np.array([self.predict(x) for x in data])
        return np.mean((true_labels == predictions).sum(axis=0) / len(true_labels))

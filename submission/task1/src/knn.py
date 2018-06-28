import numpy as np

pred_out_file = '../output/predictions.txt'


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
        dist = np.array(
            [hamming2(x[-self.input_num_bits:], y[-self.input_num_bits:]) for y in self.X])
        k_nearest = np.argsort(dist)[:self.k]
        return k_nearest

    def score(self, data, true_labels):
        predictions = np.array([self.predict(x) for x in data])
        g = (1 - (hamming2(x, y) / len(x)) for x, y in zip(predictions, true_labels))
        return sum(g) / len(true_labels)


def draw(data, seq_len, predict_len):
    train_data = data
    train_data = np.array([s[-seq_len:] for s in train_data])

    train_labels = data
    train_labels = np.array([s[-predict_len:] for s in train_labels])

    return train_data, train_labels


if __name__ == '__main__':

    d = np.loadtxt('human.txt', dtype=str)
    A = []
    for x in d:
        A.append(x)
    A = np.array(A)
    m = min(map(str.__len__, A))
    seq, end_seq = draw(A, m, 20)

    k = 7
    knn_res = []
    test_size = 100
    ps = []
    p = knn(k, 20, 30)
    p.fit(seq, end_seq)

    with open(pred_out_file, 'w') as file:
        for x in A:
            z1 = p.predict(x)
            file.write(z1 + '\n')

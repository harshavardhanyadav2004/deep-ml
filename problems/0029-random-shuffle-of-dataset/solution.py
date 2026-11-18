import numpy as np

def shuffle_data(X, y, seed=None):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(X))
    return X[shuffled_indices] , y[shuffled_indices]
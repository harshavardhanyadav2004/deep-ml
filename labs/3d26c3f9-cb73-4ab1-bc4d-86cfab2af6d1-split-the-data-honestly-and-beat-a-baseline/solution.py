# --- cell 1 ---
import numpy as np

def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    size = len(X)
    shuffled = np.random.default_rng(seed).permutation(size)
    train_size = int(size * train_frac)
    valid_size = int(size * val_frac)
    test_size = int(size * test_frac)
    train_idx = shuffled[:train_size]
    val_idx = shuffled[train_size:train_size+valid_size]
    test_idx = shuffled[train_size+valid_size:train_size+valid_size+test_size]
    train_data = X[train_idx]
    train_label = y[train_idx]
    validation_data = X[val_idx]
    validation_label = y[val_idx]
    test_data = X[test_idx]
    test_predictions = np.full(len(test_data), np.mean(train_label))
    test_label = y[test_idx]
    return test_predictions, train_idx, val_idx, test_idx
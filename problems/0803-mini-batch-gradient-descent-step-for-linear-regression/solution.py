import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    sample_data = X[batch_indices]
    mse = 2*(np.transpose(sample_data).dot(((sample_data.dot(weights))+bias)-y[batch_indices]))/(len(batch_indices))
    bias_grad = (2/len(batch_indices))*np.sum(((sample_data.dot(weights))+bias)-y[batch_indices])

    weights = weights - lr*mse
    bias = bias - lr*bias_grad

    weights = np.append(weights,bias)
    return weights
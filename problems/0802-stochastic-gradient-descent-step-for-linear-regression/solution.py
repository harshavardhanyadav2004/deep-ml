import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    n,m = X.shape
    for i in range(n_iter):
        sample =X[i%n]
        target = y[i%n]
        mse = 2*sample.dot((((sample.dot(weights))-target)))
        weights = weights - learning_rate * mse 
    return weights
        
      

import numpy as np
import math
from sklearn.metrics import log_loss

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:

    X = np.column_stack((np.ones(len(X)), X))
    predictions = []
    losses = []
    bias = np.zeros(X.shape[1])

    for i in range(iterations):
        predictions = []
        for j in range(len(X)):
            result = np.transpose(X[j]).dot(bias)
            sigmoid = 1 / (1 + math.exp(-result))
            predictions.append(sigmoid)
        predictions = np.array(predictions)
        loss = -np.sum(
            y * np.log(predictions) +
            (1 - y) * np.log(1 - predictions)
        )
        losses.append(loss)
        gradient = np.transpose(X).dot(predictions - y)
        bias = bias - learning_rate * gradient

    coeff = bias.tolist()
    coeff = [round(x, 4) for x in coeff]
    losses = [round(x, 4) for x in losses]

    return coeff, losses
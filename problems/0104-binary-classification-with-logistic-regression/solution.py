import numpy as np
import math
def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    result = []
    for elements in X:
        z = weights.dot(elements) + bias
        sigmoid = 1/(1+math.pow(math.e,-z))
        if sigmoid < 0.5 : 
            result.append(0)
        else:
            result.append(1)
    return result
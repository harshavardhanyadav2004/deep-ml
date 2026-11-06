import numpy as np
from scipy.linalg import inv
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	theta = inv(np.transpose(X).dot(np.array(X))).dot(np.transpose(X).dot(y))
	return theta
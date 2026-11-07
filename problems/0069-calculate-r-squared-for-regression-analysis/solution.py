
import numpy as np
from sklearn.metrics import r2_score as mse
def r_squared(y_true, y_pred):
	minimum = mse(y_true,y_pred)
	return minimum
	

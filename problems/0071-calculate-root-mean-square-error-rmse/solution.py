
import numpy as np
import math
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
	rmse_res = math.sqrt(mean_squared_error(y_pred,y_true))
	return round(rmse_res,3)

import numpy as np
from sklearn.metrics import mean_absolute_error
def mae(y_true, y_pred):
    absolute = mean_absolute_error(y_true,y_pred)
    return absolute
   
import numpy as np
from sklearn.metrics import precision_score as pre
def precision(y_true, y_pred):
    prec = pre(y_true,y_pred)
    return prec
import numpy as np
from sklearn.metrics import precision_score as pre
from sklearn.metrics import recall_score as rec
def f_score(y_true, y_pred, beta):
    result = pre(y_true,y_pred)
    result_one = rec(y_true,y_pred)
    denom = (beta*beta*result+result_one)
    if denom==0:
        return 0.0
    f_beta = (1+beta*beta)*(result*result_one)/denom
    return round(f_beta,3)
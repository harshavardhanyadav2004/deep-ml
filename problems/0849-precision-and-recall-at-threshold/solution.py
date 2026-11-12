import numpy as np
from sklearn.metrics import precision_score as result
from sklearn.metrics import recall_score as res
def precision_recall_at_threshold(y_true, y_scores, threshold):
    y_pred = []
    for score in y_scores:
        if score >= threshold:
            y_pred.append(1)
        else: y_pred.append(0)
    pre = result(y_true,y_pred)
    rec = res(y_true,y_pred)
    if not any(y_pred) : pre = 0.0
    if  not any(y_true) : rec = 0.0
    return [round(pre,4),round(rec,4)]
    
   

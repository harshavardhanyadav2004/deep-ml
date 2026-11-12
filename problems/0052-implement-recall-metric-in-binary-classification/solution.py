import numpy as np
from sklearn.metrics import recall_score as call
def recall(y_true, y_pred):
    result = call(y_true,y_pred)
    return result 


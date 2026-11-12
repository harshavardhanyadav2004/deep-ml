import numpy as np
from sklearn.metrics import accuracy_score as acc
def accuracy_score(y_true, y_pred):
    accuracy = acc(y_true,y_pred)
    return accuracy
    
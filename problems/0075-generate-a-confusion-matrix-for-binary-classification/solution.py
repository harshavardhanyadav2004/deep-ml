
from collections import Counter
from sklearn.metrics import confusion_matrix as cf
import numpy as np
def confusion_matrix(data):
    data = np.transpose(data)
    generate = cf(data[0],data[1])
   
    elements = [list(i)[::-1] for i in generate]
    
    return elements[::-1]
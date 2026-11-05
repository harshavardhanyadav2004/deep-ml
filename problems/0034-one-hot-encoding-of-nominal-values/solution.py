import numpy as np
from sklearn.preprocessing import OneHotEncoder
def to_categorical(x, n_col=None):
	possible = []
	if n_col is  not None :
		possible = [ i for i in range(n_col)]
	else: 
		possible = [ i for i in set(x)]
	array = np.array(x)
	x = array.reshape(-1,1)
	onehotencoder = OneHotEncoder(
		categories = [possible] ,
		sparse_output = False)
	data = onehotencoder.fit_transform(x)
	return data 
	
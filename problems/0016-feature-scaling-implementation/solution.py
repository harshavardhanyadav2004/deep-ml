import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	normalized_data , standardized_data = [],[]
	data = np.transpose(data)
	for array in data:
		one,two = [],[]
		for element in array:
			result_one = ((element-np.mean(array))/np.std(array))
			two.append(round(result_one,4))
			result = ((element-min(array))/(max(array)-min(array)))
			one.append(round(result,4))
		standardized_data.append(two)
		normalized_data.append(one)
	normalized_data = np.transpose(normalized_data)
	standardized_data = np.transpose(standardized_data)
	return standardized_data, normalized_data
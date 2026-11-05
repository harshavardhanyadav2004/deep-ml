import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    data = np.transpose(X_train)
    std_array , mean_array = [],[]
    for column in data:
        std_array.append(np.std(column))
        mean_array.append(np.mean(column))
    new_data = np.transpose(X_test)
    result = []
    for i in range(len(new_data)):
        another = []
        for j in range(len(new_data[0])):
            standard_deviation = std_array[i]
            if standard_deviation == 0:
                standard_deviation = 1.0
            another.append((new_data[i][j]-mean_array[i])/standard_deviation)
        result.append(another)
    result = np.transpose(result)
    return result    
           



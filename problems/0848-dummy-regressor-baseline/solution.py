import numpy as np
from sklearn.dummy import DummyRegressor
def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    x_train = np.zeros((len(y_train),1))
    x_test = np.zeros(n_test)
    regressor = DummyRegressor(strategy = strategy , constant = constant , quantile =quantile)
    regressor.fit(x_train,y_train)
    return regressor.predict(x_test)
    


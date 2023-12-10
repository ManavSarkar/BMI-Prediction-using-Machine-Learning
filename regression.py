from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process.kernels import Exponentiation,RationalQuadratic
import pandas as pd
import sklearn.gaussian_process 
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


def RMSE(MSE):
    return np.sqrt(MSE)

train_df = pd.read_csv('Image_train.csv',header=None)
X_train, y_train = train_df.iloc[:,2:], train_df.iloc[:,1]

test_df = pd.read_csv('Image_test.csv',header=None)
X_test, y_test = test_df.iloc[:,2:], test_df.iloc[:,1]

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


KN = Exponentiation(RationalQuadratic(), exponent=2)
gpr = sklearn.gaussian_process.GaussianProcessRegressor(kernel=KN, alpha=1e-3)


gpr.fit(X_train, y_train)


y_pred = gpr.predict(X_test)

print('MAE: ', mean_absolute_error(y_test, y_pred),
              'MAPE: ', mean_absolute_percentage_error(y_test, y_pred),
              'R2: ', r2_score(y_test, y_pred),
              'RMSE: ', RMSE(mean_squared_error(y_test, y_pred)),
              )
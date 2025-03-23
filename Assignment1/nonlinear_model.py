import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

Training_data = pd.read_csv('Data4Regression - Training Data.csv', usecols=[0, 1], header=None, skiprows=1)
Test_data = pd.read_csv('Data4Regression - Test Data.csv', usecols=[0,1], header=None, skiprows=1)

Training_x = Training_data[0].to_numpy().reshape(-1, 1)
Training_y = Training_data[1].to_numpy().reshape(-1, 1)

Test_x = Test_data[0].to_numpy().reshape(-1, 1)
Test_y = Test_data[1].to_numpy().reshape(-1, 1)

Training_x = np.hstack((np.ones((Training_x.shape[0], 1)), Training_x))
Test_x = np.hstack((np.ones((Test_x.shape[0], 1)), Test_x))

sigma = 0.1
lambda_reg = 0.6

# 高斯核函数
Training_K = rbf_kernel(Training_x, Training_x, gamma=1 / (2 * sigma ** 2))
Test_K = rbf_kernel(Test_x, Test_x, gamma=1 / (2 * sigma ** 2))

alpha = np.linalg.inv(Training_K + lambda_reg * np.eye(len(Training_x))) @ Training_y

y_pred = Training_K @ alpha
y_pred_test = Test_K @ alpha

def Error(y, y_pred):
    error = 1/2 * (1 / y.shape[0]) * np.sum((y - y_pred) ** 2)
    return error

error_train = Error(Training_y, y_pred)
print(error_train)
error_test = Error(Test_y, y_pred_test)
print(error_test)
plt.scatter(Training_x[:, 1], Training_y, color='blue', label='Original data')
plt.plot(Training_x[:, 1], y_pred, color='red', label='Gaussian kernel fit')
plt.title('Gaussian kernel')
plt.savefig('nonlinear.png')

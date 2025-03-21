import numpy as np
import pandas as pd
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

# pdb.set_trace()

def OLS(x, y):
    theta = (np.linalg.inv(x.T @ x)) @ x.T @ y
    return theta

def GD(x, y, iterations=200, lr=0.05):
    theta = np.ones(x.shape[1]).reshape(-1, 1)
    for i in range(iterations):
        grad_theta = (1 / x.shape[0]) * x.T @ (x @ theta - y)
        theta = theta - lr * grad_theta
    return theta

def NT(x, y, iterations=10):
    theta = np.ones(x.shape[1]).reshape(-1, 1)
    for i in range(iterations):
        grad_theta = x.T @ (x @ theta - y)
        hessian = x.T @ x
        theta = theta - np.linalg.inv(hessian) @ grad_theta
    return theta

def Error(x, y, theta):
    error = 1/2 * (1 / x.shape[0]) * np.sum((x @ theta - y) ** 2)
    return error

theta_OLS = OLS(Training_x, Training_y)
print(theta_OLS)
OLS_Training_error = Error(Training_x, Training_y, theta_OLS)
OLS_Test_error = Error(Test_x, Test_y, theta_OLS)
print(OLS_Training_error, OLS_Test_error)

theta_GD = GD(Training_x, Training_y)
print(theta_GD)
GD_Training_error = Error(Training_x, Training_y, theta_GD)
GD_Test_error = Error(Test_x, Test_y, theta_GD)
print(GD_Training_error, GD_Test_error)

theta_NT = NT(Training_x, Training_y)
print(theta_NT)
NT_Training_error = Error(Training_x, Training_y, theta_NT)
NT_Test_error = Error(Test_x, Test_y, theta_NT)
print(NT_Training_error, NT_Test_error)

def Show(theta):
    plt.scatter(Training_x[:, 1], Training_y, color='blue', label='Training_data')
    plt.plot(Training_x[:, 1], Training_x @ theta, 'r', label='Fitted')

Show(theta_OLS)
plt.title("OLS")
plt.savefig('OLS.png')
plt.close()

Show(theta_GD)
plt.title("GD")
plt.savefig('GD.png')
plt.close()

Show(theta_NT)
plt.title("NT")
plt.savefig('NT.png')
plt.close()
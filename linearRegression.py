import random

import numpy as np
from matplotlib import pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from random import uniform
from sklearn.datasets import fetch_california_housing
from scipy import optimize as op
from sklearn.datasets import make_regression
def linearRegression(X, y):
    m, n = X.shape # m = number of examples, n = number of features
    theta_ini = np.zeros(n + 1) # theta_0 is the bias variable
    theta_ini[0] = 1 # convention is that theta_0 = 1
    X_bias = np.insert(X, 0, np.ones(X.shape[0]), axis=1) # adding column of ones for the bias variable

    # gradient descent
    options= {'maxiter': 400}
    # jac=true if costfunction also returns gradient
    theta = op.minimize(costFunction, theta_ini, (X_bias,y), method='TNC').x
    return theta

def costFunction(theta, X, y, lambda_=0.0):
    m,n = X.shape
    return (np.sum(np.square(np.matmul(X, theta) - y)) + lambda_ * np.sum(np.square(theta[1:])))/(2*m)

def main():
    #X, y = fetch_california_housing(return_X_y=True)
    bias = random.uniform(-10,10)
    X, y, coef = make_regression(n_samples=150, n_features=1, noise=10, bias=bias, coef=True)
    linearRegression(X, y)
    model = LinearRegression().fit(X, y)
    theta_model = np.insert(model.coef_, 0, model.intercept_)
    theta = linearRegression(X, y)
    print(coef)
    print(theta_model)
    print(theta)
    print('done')

if __name__ == "__main__":
    main()
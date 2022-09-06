import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from random import uniform
from sklearn.datasets import fetch_california_housing
from scipy import optimize as op
from sklearn.datasets import make_regression
def linearRegression(df):
    X = df.data.to_numpy()
    y = df.target.to_numpy()
    m, n = X.shape # m = number of examples, n = number of features
    theta_ini = np.zeros(n + 1) # theta_0 is the bias variable
    theta_ini[0] = 1 # convention is that theta_0 = 1
    X_bias = np.insert(X, 0, np.ones(X.shape[0]), axis=1) # adding column of ones for the bias variable

    # gradient descent
    # method 'BFGS' returns hypothesis function close to sklearn's hypothesis
    # method ‘Newton-CG’
    theta = op.minimize(costFunction, theta_ini, (X_bias,y), method='BFGS').x
    return theta

def costFunction(theta, X, y, lambda_=0.0):
    m,n = X.shape
    return (np.sum(np.square(np.matmul(X, theta) - y)) + lambda_ * np.sum(np.square(theta[1:])))/(2*m)

def main():
    df = fetch_california_housing(as_frame=True)
    theta = linearRegression(df)
    skTheta = LinearRegression().fit(df.data.to_numpy(), df.target.to_numpy())
    skTheta = np.insert(skTheta.coef_, 0, skTheta.intercept_)
    X = df.data.to_numpy()
    y = df.target.to_numpy()
    X_bias = np.insert(X, 0, np.ones(X.shape[0]), axis=1)  # adding column of ones for the bias variable
    print('My implementation\'s hypothesis function:', theta)
    print('MSE of my implementation: ', mean_squared_error(y, np.matmul(X_bias, skTheta)))
    print('skLearn\'s implementation\'s hypothesis function:',skTheta)
    print('MSE of sklearn\'s implimentation: ', mean_squared_error(y, np.matmul(X_bias, skTheta)))

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy import optimize as op

def logisticRegression(X, y, classes):
    m, n = X.shape  # m = number of examples, n = number of features, not including bias
    theta = {}
    for currClass in classes:
        yCurr = (y == currClass)
        yCurr = yCurr.astype(int)
        theta_ini = np.zeros(n + 1)  # theta_0 is the bias variable
        theta_ini[0] = 1  # convention is that theta_0 = 1
        X_bias = np.insert(X, 0, np.ones(X.shape[0]), axis=1)  # adding column of ones for the bias variable
        # gradient descent
        currTheta = op.minimize(costFunction, theta_ini, (X_bias, yCurr), method='BFGS').x
        theta[currClass] = currTheta
    return theta

def costFunction(theta, X, y, lambda_=0.0):
    m = X.shape[0]
    yp = expit(np.matmul(X, theta))
    eps = 1e-5
    leftOp = np.sum(y * np.log(yp + eps))
    rightOp = np.sum((1 - y) * np.log(1 - yp + eps))
    # reg = np.sum(np.square(theta[1:])) * lambda_ / 2
    cost = -1*(leftOp + rightOp) / m
    return cost

def main():
    df = load_penguins()
    # Data cleaning:
    # There are columns with NaN, so we drop those
    df = df.dropna()
    # Dataset has island, sex, year, all of which are not needed so we drop these variables
    features = df.drop(columns=['island', 'sex', 'year'])
    target = features.drop(columns=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    features = features.drop(columns='species')

    # Convert features and target to numpy arrays
    X = features.to_numpy()
    y = target.to_numpy()
    y = y.reshape(y.shape[0])
    species = np.unique(y)

    theta = logisticRegression(X, y, species)
    model =  LogisticRegression(max_iter=1000).fit(X,y)
    # print(model.classes_)
    # print(model.coef_)
    # print(model.intercept_)
    skTheta = np.insert(model.coef_, 0, model.intercept_, axis=1)
    print(theta)
    print(skTheta)
if __name__ == "__main__":
    main()
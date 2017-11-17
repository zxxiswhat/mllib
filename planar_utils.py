import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y):
    # set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[0, :].min() - 1, X[0, :].max() + 1
    h = 0.01
    # Generate a grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    # xx.ravel() xx=[(1,2,3),(4,5,6)]  xx.ravel() = [1,2,3,4,5,6], but chanel xx.ravel() will change xx self
    # np.c_: a=[(1,2,3)] b = [(7,8,9)] => c=[(1,7),(2,8),(3,9)]
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # plot
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y,cmap=plt.cm.Spectral)


def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s

def load_planar_dataset():
    np.random.seed(1)
    m = 400
    N = int(m / 2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j, N*(j+1))
        t = np.linspace(j*3.12, (j+1)*3.12, N) + np.random.randn(N) * 0.2
        r = a*np.sin(4*t) + np.random.randn(N)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j


    X = X.T
    Y = Y.T

    return X, Y
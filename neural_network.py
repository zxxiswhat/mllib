import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0, :], cmap=plt.cm.Spectral)

# set the hidden layers unit count
def layer_size(X, Y, hidden_unit):
    n_x = X.shape[0]
    n_h = hidden_unit
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

# forward algorithm
def initialize_parameter(n_x, n_h, n_y):
    np.random.seed(2);
    # WX+B W1:(n_h, n_x) b1:(n_h, n_x) W2:(n_y, n_h) b2:(n_y, n_h)
    # why multipy 0.01
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {'W1': W1, 'b1':b1, 'W2': W2, 'b2': b2}
    return parameters;

def forward_propogation(X, parameters):
    # X(n_x, 400) W1(n_h,n_x)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    #A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {'Z1':Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache;

def cost_funciton(A2, Y):
    # Y(1, 400) A2(1, 400)
    m = Y.shape[1]
    J = -1 * ((np.dot(np.log(A2), Y.T) + np.dot(np.log(1 - A2), (1 - Y).T)) / m)
    #J = np.squeeze(J)
    return J

def backward_propagation(parameters, cache, X, Y):
    # Y(1, 400) A2(1, 400)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    A2 = cache['A2']
    Z2 = cache['Z2']
    A1 = cache['A1']
    Z1 = cache['Z1']

    m = X.shape[1]

    #A1(hideen_unit ,400)
    dZ2 = A2 - Y
    # dZ2(1, 400)
    # w2 (1, hideen_unit)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims=True) / m


    # (1, hidden_unit), (1,400) ＝ (hidden_unit， 400)
    # (hidden_unit， 400) * (hidden_unit， 400).T * (hidden_unit, 400)

    # dZ1(hidden_unit, 400)
    dZ1 = (np.dot(W2.T, dZ2) * A1 *(1 - A1))
    #dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims=True) / m


    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return parameters


def train_model(X, Y, n_h, num_iterations = 10000, print_cost = False):
    np.random.seed(3)
    n_x, n_h, n_y = layer_size(X, Y, 4)
    parameters = initialize_parameter(n_x, n_h, n_y)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    for i in range(0, num_iterations):

        A2, cache = forward_propogation(X, parameters)

        J = cost_funciton(A2, Y)

        if print_cost and i % 1000 == 0:
            print("interater %i: %f" %(i, J))

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)


    return parameters



def predict(parameters, X):
    A2, cache = forward_propogation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions


np.random.seed(1)

X, Y = load_planar_dataset()

### START CODE HERE ### (≈ 3 lines of code)
shape_X = X.shape;
shape_Y = Y.shape;
### END CODE HERE ###



plt.scatter(X[0, :], X[1, :], c=Y[0, :], s = 40, cmap=plt.cm.Spectral)

print("shape_x:", shape_X)
print("shape_y:", shape_Y)

parameters = train_model(X, Y, n_h = 4, num_iterations = 10000, print_cost = True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))




plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
right = float((np.dot(Y, predictions.T) + np.dot((1 - Y), (1 - predictions).T)) / float(Y.size) * 100)
print ("accurancy:", right)

# 要注意的事宜：
# 1. 首先, W b A Z的维度区别：A和Z的维度和sample的个数相同，因此在计算的时候不用取均值，直接用向量。但是W 和b的维度和sample无关，因此计算的时候需要取均值
# 2. np.sum(dZ2, axis = 1, keepdims=True)
# lambda函数使用：lambda x（参数）: predict(parameters, x.T)（动作）
#

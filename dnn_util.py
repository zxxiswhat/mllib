import numpy as np
import h5py

def sigmoid_forward(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = [Z, A]
    return A, cache

def relu_forward(Z):
    A = np.maximum(0, Z)
    cache = [Z, A]
    return A, cache

def leak_relu_forward(Z, threadhold):
    A = np.maximum(threadhold, Z)
    cache = [Z, A]
    return A, cache

def tanh_forward(Z):
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    cache = [Z, A]
    return A, cache

def soft_max_forward(Z):
    ZExp = np.exp(Z)
    SumExp = np.sum(ZExp, axis=0, keepdims=True)
    A = ZExp / SumExp
    cache = [Z, A]
    return A, cache

def sigmoid_backforward(A):
    dZ = A * (1 - A)
    return dZ

def relu_backforward(A):
    dZ = np.array(A, copy=True)
    dZ[A > 0] = 1
    return dZ

def leak_relu_backforward(A):
    dZ = np.array(A, copy=True)
    dZ[A > 0] = 1
    return dZ

def tanh_backforward(A):
    dZ = 1 - np.power(A, 2)
    return dZ

def soft_max_backforward(A):
    dZ = np.ones(A.shape)
    return dZ

def load_data():
    train_dataset = h5py.File('/Users/baidu/Desktop/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('/Users/baidu/Desktop/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
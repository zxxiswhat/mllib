import numpy as np
import dnn_util as du

def initialize_deep_parameters(layer_dims, func="he", batch_normalization=False):
    np.random.seed(1)

    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        # this one
        # parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l-1])
        # he initialize
        if func=="he":
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        elif func=="random":
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.001
        elif func=="tanh":
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(1 / layer_dims[l - 1])
        elif func=="ys":
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / (layer_dims[l - 1] + layer_dims[l]))

        if batch_normalization:
            parameters['gamma' + str(l)] = np.ones((layer_dims[l], 1))
            parameters['beta' + str(l)] = np.zeros((layer_dims[l], 1))
            parameters['mu_mean' + str(l)] = np.ones((layer_dims[l], 1))
            parameters['enta2_mean' + str(l)] = np.zeros((layer_dims[l], 1))

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def getL(parameters, batch_normalization=False):
    if batch_normalization:
        return len(parameters)//6
    else:
        return len(parameters)//2

def forward_linear_block(W, A, b, DROPOUT=False):
    cache = (W, A, b)
    Z = np.dot(W, A) + b
    return Z, cache


# cache = (linear_cache, activation_cache)
# linear_cache:(W, A, b)
# activation_cache: (Z, A, D)
def forward_block(W, A_prev, b, activation, keep_prob=1.0, epsilon=1e-8):

    Z, linear_cache = forward_linear_block(W, A_prev, b)

    if activation == "sigmoid":
        A, activation_cache = du.sigmoid_forward(Z)
    elif activation == "relu":
        A, activation_cache = du.relu_forward(Z)
    elif activation == "leak_relu":
        A, activation_cache = du.leak_relu_forward(Z, 0.01)
    elif activation == "tanh":
        A, activation_cache = du.tanh_forward(Z)
    elif activation == "softmax":
        A, activation_cache = du.soft_max_forward(Z)


    D = np.random.rand(A.shape[0], A.shape[1]) <= keep_prob
    activation_cache.append(D)
    A = np.multiply(A, D)
    A /= keep_prob

    cache = (linear_cache, activation_cache)
    return A, cache

# cache = (linear_cache, activation_cache)
# linear_cache:(W, A_pre, b)
# activation_cache: (Z~, A, Z, Znorm, mu, enta2, beta, gamma, D, mu_mean, enta2_mean)
def forward_batch_nomalization_block(W, A_prev, b, activation, beta, gamma, mu_mean, enta2_mean, mu_enta2_mean_beta=0.9, keep_prob=1.0, epsilon=1e-8):

    Z, linear_cache = forward_linear_block(W, A_prev, b)


    mu = np.mean(Z, axis=1).reshape((Z.shape[0], 1))
    enta2 = np.var(Z, axis=1).reshape((Z.shape[0], 1))
    Znorm = np.divide((Z - mu), np.sqrt(enta2 + epsilon))
    Zbo = np.multiply(gamma, Znorm) + beta

    mu_mean_new = mu_enta2_mean_beta * mu_mean + (1 - mu_enta2_mean_beta) * mu
    enta2_mean_new = mu_enta2_mean_beta * enta2_mean + (1 - mu_enta2_mean_beta) * enta2

    if activation == "sigmoid":
        A, activation_cache = du.sigmoid_forward(Zbo)
    elif activation == "relu":
        A, activation_cache = du.relu_forward(Zbo)
    elif activation == "leak_relu":
        A, activation_cache = du.leak_relu_forward(Zbo, 0.01)
    elif activation == "tanh":
        A, activation_cache = du.tanh_forward(Zbo)
    elif activation == "softmax":
        A, activation_cache = du.soft_max_forward(Z)

    activation_cache.append(Z)
    activation_cache.append(Znorm)
    activation_cache.append(mu)
    activation_cache.append(enta2)
    activation_cache.append(beta)
    activation_cache.append(gamma)


    D = np.random.rand(A.shape[0], A.shape[1]) <= keep_prob
    activation_cache.append(D)
    A = np.multiply(A, D)
    A /= keep_prob

    activation_cache.append(mu_mean_new)
    activation_cache.append(enta2_mean_new)

    cache = (linear_cache, activation_cache)
    return A, cache

def activation_backward(A, activation):

    if activation == "sigmoid":
        g = du.sigmoid_backforward(A)
    elif activation == "relu":
        g = du.relu_backforward(A)
    elif activation == "leak_relu":
        g = du.leak_relu_backforward(A)
    elif activation == "tanh":
        g = du.tanh_backforward(A)
    elif activation == "softmax":
        g = du.soft_max_backforward(A)
    return g

def backward_block(dA, cache, activation, regularation="L2", lamd=0.99, keep_prob=1.0):

    linear_cache = cache[0]
    activation_cache = cache[1]

    W = linear_cache[0]
    A_prev = linear_cache[1]
    b = linear_cache[2]

    Z = activation_cache[0]
    A = activation_cache[1]
    D = activation_cache[2]

    g = activation_backward(A, activation)

    m = dA.shape[1]
    dZ = dA * g
    dW = np.dot(dZ, A_prev.T).reshape(W.shape) / m
    if regularation=="L2":
        dW = dW + (lamd / m) * W


    db = np.sum(dZ, axis=1, keepdims=True).reshape(b.shape) / m
    dA_pre = np.dot(W.T, dZ)

    return dW, db, dA_pre

# activation_cache: (Z~, A, Z, Znorm, mu, enta2, beta, gamma, D)
# grad:
def backward_batch_nomalization_block(dA, cache, activation, epsilon=1e-8, regularation="L2", lamd=0.99, keep_prob=1.0):

    linear_cache = cache[0]
    activation_cache = cache[1]

    W = linear_cache[0]
    A_prev = linear_cache[1]
    b = linear_cache[2]

    Zbo = activation_cache[0]
    A = activation_cache[1]
    Z = activation_cache[2]
    Znorm = activation_cache[3]
    mu = activation_cache[4]
    enta2 = activation_cache[5]
    beta = activation_cache[6]
    gamma = activation_cache[7]
    D = activation_cache[8]

    g = activation_backward(A, activation)

    std_inv = 1. / np.sqrt(enta2 + epsilon)
    X_mu = Z - mu

    m = dA.shape[1]
    dZbo = dA * g
    dgamma = np.sum(dZbo * Znorm, axis=1, keepdims=True).reshape(dZbo.shape[0], 1)
    dbeta = np.sum(dZbo, axis=1).reshape(dZbo.shape[0], 1)
    dZnorm = dZbo * gamma
    denta2 = np.sum(X_mu * dZnorm, axis=1, keepdims=True) * np.power(std_inv, 3) * -0.5
    dmu = -np.sum(dZnorm, axis=1, keepdims=True) * std_inv + denta2 * (-2) * np.mean(X_mu, axis=1, keepdims=True)
    dZ = dZnorm * std_inv + denta2 * 2 * X_mu / m + dmu / m

    dW = np.dot(dZ, A_prev.T).reshape(W.shape) / m
    if regularation=="L2":
        dW = dW + (lamd / m) * W


    db = np.sum(dZ, axis=1, keepdims=True).reshape(b.shape) / m
    dA_pre = np.dot(W.T, dZ)

    return dW, db, dA_pre, dgamma, dbeta

def dropout_keep_prob(L, drop_out=False):
    if drop_out:
        keep_prob = (0.5, 0.7, 1, 1)
    else:
        keep_prob = np.ones((L, 1))
    return keep_prob

def forward_model(parameters, X, DROPOUT=False, batch_normalization=False):
    np.random.seed(1)
    L = getL(parameters, batch_normalization)
    # print("L forward:" + str(L))
    activations = L_activations(L)
    assert ((L) == len(activations))
    caches = []
    A_prev = X


    keep_prob = dropout_keep_prob(L, drop_out=DROPOUT)

    for l in range(0, L):
        W = parameters['W' + str(l + 1)]
        b = parameters['b' + str(l + 1)]

        if batch_normalization and l != (L - 1):
            beta = parameters['beta' + str(l + 1)]
            gamma = parameters['gamma' + str(l + 1)]
            mu_mean = parameters['mu_mean' + str(l + 1)]
            enta2_mean = parameters['enta2_mean' + str(l + 1)]
            A, cache = forward_batch_nomalization_block(W=W, A_prev=A_prev, b=b, activation=activations[l], beta=beta, gamma=gamma, mu_mean=mu_mean, enta2_mean=enta2_mean, mu_enta2_mean_beta=0.9, keep_prob=keep_prob[l], epsilon=1e-8)

        else:
            A, cache = forward_block(W, A_prev, b, activations[l], keep_prob=keep_prob[l])

        A_prev = A
        caches.append(cache)

    return A_prev, caches

def backward_model(AL, Y, caches, regularation="L2", batch_normalization=False):
    grads = {}
    L = len(caches)
    # print("L back:" + str(L))
    activations = L_activations(L)

    if activations[L - 1] == "softmax":
        dA = AL - Y
    else:
        dA = -(np.divide(Y, AL)) + np.divide((1-Y), (1 - AL))

    if regularation == "DROPOUT":
        keep_prob = dropout_keep_prob(L, drop_out=True)
    else:
        keep_prob = dropout_keep_prob(L, drop_out=False)

    for l in reversed(range(L)):
        cache = caches[l]

        if regularation == "DROPOUT":
            activation_cache = cache[1]
            D = activation_cache[2]
            dA = np.multiply(dA, D)
            dA /= keep_prob[l]
            grads["dA" + str(l + 1)] = dA

        if batch_normalization and l != (L - 1):
            dW, db, dA, dgamma, dbeta = backward_batch_nomalization_block(dA=dA, cache=cache, activation=activations[l], epsilon=1e-8, regularation=regularation, lamd=0.99, keep_prob=keep_prob[l])
            grads["dgamma" + str(l + 1)] = dgamma
            grads["dbeta" + str(l + 1)] = dbeta
        else:
            dW, db, dA = backward_block(dA=dA, cache=cache, activation=activations[l], regularation=regularation, keep_prob=keep_prob[l])

        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db


    return grads

def adam_function():
    print("adam")

def cost_function(AL, Y, parameters, regularation="L2", lamd=0.1, batch_normalization=False, softmax=False):
    m = Y.shape[1]
    t = 0
    # tmp =  (1 - Y).T
    # tmp =  -np.dot(np.log(1 - AL), (1 - Y).T)
    if softmax:
        J = -np.sum(Y * np.log(AL)) / m
    else:
        J = -1 * ((np.dot(np.log(AL), Y.T) + np.dot(np.log(1 - AL), (1 - Y).T)) / m)
    np.log(AL)

    if regularation=="L2":
        L = getL(parameters, batch_normalization)
        reg_sum = 0.
        for l in range(1, L + 1):
            reg_sum = reg_sum + np.sum(np.square(parameters['W' + str(l)]))
        J = J  + reg_sum * lamd / (2 * m)

    elif regularation=="dropout":
        t = t + 1

    return J


def update_parameters(grads, parameters, learning_rate=0.01, batch_normalization=False):
    L = getL(parameters, batch_normalization)
    for l in range(1, L + 1):
        parameters['W' + str(l)] = parameters['W' + str(l)] - np.multiply(learning_rate,grads['dW' + str(l)])
        parameters['b' + str(l)] = parameters['b' + str(l)] - np.multiply(learning_rate,grads['db' + str(l)])
        if batch_normalization==True and l != L:
            parameters['beta' + str(l)] = parameters['beta' + str(l)] - np.multiply(learning_rate, grads['dbeta' + str(l)])
            parameters['gamma' + str(l)] = parameters['gamma' + str(l)] - np.multiply(learning_rate, grads['dgamma' + str(l)])

    return parameters

def update_parameters_momentum(grads, parameters, v, beta, learning_rate=0.01, batch_normalization=False):
    L = getL(parameters, batch_normalization)

    for l in range(1, L + 1):
        v['dW' + str(l)] = beta * v['dW' + str(l)] + (1 - beta) * grads['dW' + str(l)]
        v['db' + str(l)] = beta * v['db' + str(l)] + (1 - beta) * grads['db' + str(l)]

        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * v['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * v['db' + str(l)]

        if batch_normalization == True and l != L:
            v['dbeta' + str(l)] = beta * v['dbeta' + str(l)] + (1 - beta) * grads['dbeta' + str(l)]
            v['dgamma' + str(l)] = beta * v['dgamma' + str(l)] + (1 - beta) * grads['dgamma' + str(l)]
            parameters['beta' + str(l)] = parameters['beta' + str(l)] - learning_rate * v['dbeta' + str(l)]
            parameters['gamma' + str(l)] = parameters['gamma' + str(l)] - learning_rate * v['dgamma' + str(l)]


    return parameters, v

def update_parameters_rmsprop(grads, parameters, s, beta, epsilon, learning_rate=0.01, batch_normalization=False):
    L = getL(parameters, batch_normalization)

    for l in range(1, L + 1):
        s['dW' + str(l)] = beta * s['dW' + str(l)] + (1 - beta) * np.power(grads['dW' + str(l)], 2)
        s['db' + str(l)] = beta * s['db' + str(l)] + (1 - beta) * np.power(grads['db' + str(l)], 2)

        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)] / (np.sqrt(s['dW' + str(l)]) + epsilon)
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads['db' + str(l)] / (np.sqrt(s['db' + str(l)]) + epsilon)

        if batch_normalization == True  and l != L:
            s['dbeta' + str(l)] = beta * s['dbeta' + str(l)] + (1 - beta) * np.power(grads['dbeta' + str(l)], 2)
            s['dgamma' + str(l)] = beta * s['dgamma' + str(l)] + (1 - beta) * np.power(grads['dgamma' + str(l)], 2)

            parameters['beta' + str(l)] = parameters['beta' + str(l)] - learning_rate * grads['dbeta' + str(l)] / (np.sqrt(s['dbeta' + str(l)]) + epsilon)
            parameters['gamma' + str(l)] = parameters['gamma' + str(l)] - learning_rate * grads['dgamma' + str(l)] / (np.sqrt(s['dgamma' + str(l)]) + epsilon)

    return parameters, s


def update_parameters_adam(grads, parameters, v, s, t, epsilon=1e-8, beta1=0.9, beta2=0.999, learning_rate=0.01, batch_normalization=False):
    L = getL(parameters, batch_normalization)
    v_c = {}
    s_c = {}
    for l in range(1, L + 1):
        v['dW' + str(l)] = beta1 * v['dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        v_c['dW' + str(l)] = v['dW' + str(l)] / (1 - np.power(beta1, t))
        v['db' + str(l)] = beta1 * v['db' + str(l)] + (1 - beta1) * grads['db' + str(l)]
        v_c['db' + str(l)] = v['db' + str(l)] / (1 - np.power(beta1, t))
        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1 - beta2) * np.power(grads['dW' + str(l)], 2)
        s_c['dW' + str(l)] = s['dW' + str(l)] / (1 - np.power(beta2, t))
        s['db' + str(l)] = beta2 * s['db' + str(l)] + (1 - beta2) * np.power(grads['db' + str(l)], 2)
        s_c['db' + str(l)] = s['db' + str(l)] / (1 - np.power(beta2, t))
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * v_c['dW' + str(l)] / (np.sqrt(s_c['dW' + str(l)]) + epsilon)
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * v_c['db' + str(l)] / (np.sqrt(s_c['db' + str(l)]) + epsilon)

        if batch_normalization == True and l != L:
            v['dbeta' + str(l)] = beta1 * v['dbeta' + str(l)] + (1 - beta1) * grads['dbeta' + str(l)]
            v_c['dbeta' + str(l)] = v['dbeta' + str(l)] / (1 - np.power(beta1, t))
            v['dgamma' + str(l)] = beta1 * v['dgamma' + str(l)] + (1 - beta1) * grads['dgamma' + str(l)]
            v_c['dgamma' + str(l)] = v['dgamma' + str(l)] / (1 - np.power(beta1, t))
            s['dbeta' + str(l)] = beta2 * s['dbeta' + str(l)] + (1 - beta2) * np.power(grads['dbeta' + str(l)], 2)
            s_c['dbeta' + str(l)] = s['dbeta' + str(l)] / (1 - np.power(beta2, t))
            s['dgamma' + str(l)] = beta2 * s['dgamma' + str(l)] + (1 - beta2) * np.power(grads['dgamma' + str(l)], 2)
            s_c['dgamma' + str(l)] = s['dgamma' + str(l)] / (1 - np.power(beta2, t))
            parameters['beta' + str(l)] = parameters['beta' + str(l)] - learning_rate * v_c['dbeta' + str(l)] / (np.sqrt(s_c['dbeta' + str(l)]) + epsilon)
            parameters['gamma' + str(l)] = parameters['gamma' + str(l)] - learning_rate * v_c['dgamma' + str(l)] / (np.sqrt(s_c['dgamma' + str(l)]) + epsilon)

    return parameters, v, s

def L_activations(activations_len):
    # print("L activations_len:" + str(activations_len))

    activations = []


    for l in range(0, activations_len - 1):
        activations.append("relu")

    activations.append("sigmoid")
    #activations.append("softmax")
    # print("L activations_len:" + str(len(activations)))
    return activations

def initialize_optimizer_momentum(layer_dims, batch_normalization=False):
    L = len(layer_dims)
    v = {}
    for l in range(1, L):
        v["dW" + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1]))
        v["db" + str(l)] = np.zeros((layer_dims[l], 1))
        if batch_normalization:
            v["dbeta" + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1]))
            v["dgamma" + str(l)] = np.zeros((layer_dims[l], 1))

    return v


def initialize_optimizer_rmsprop(layer_dims, batch_normalization=False):
    L = len(layer_dims)
    s = {}
    for l in range(1, L):
        s["dW" + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1]))
        s["db" + str(l)] = np.zeros((layer_dims[l], 1))
        if batch_normalization:
            s["dbeta" + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1]))
            s["dgamma" + str(l)] = np.zeros((layer_dims[l], 1))

    return s

def initialize_optimizer_adam(layer_dims, batch_normalization=False):
    v = initialize_optimizer_momentum(layer_dims)
    s = initialize_optimizer_rmsprop(layer_dims)
    return v,s


def L_layer_model(X, Y, layer_dims, optimizer="gd", regularation="L2", init_weight_func="tanh", beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate = 0.0075, num_iterations = 10000, batch_normalization=False, print_cost=False):

    t = 1
    parameters = initialize_deep_parameters(layer_dims, init_weight_func, batch_normalization=batch_normalization)
    L = getL(parameters, batch_normalization)
    activations = L_activations(L)
    if activations[L - 1]=="softmax":
        softmax = True
    else:
        softmax = False
    if optimizer == "momentum":
        v = initialize_optimizer_momentum(layer_dims, batch_normalization=batch_normalization)
    elif optimizer == "rmsprop":
        s = initialize_optimizer_rmsprop(layer_dims, batch_normalization=batch_normalization)
    elif optimizer == "adam":
        v, s = initialize_optimizer_adam(layer_dims, batch_normalization=batch_normalization)

    for i in range(0, num_iterations):
        # AL, caches = forward_model(parameters, X, DROPOUT=True)
        AL, caches = forward_model(parameters, X, DROPOUT=False, batch_normalization=batch_normalization)

        J = cost_function(AL, Y, parameters, regularation, batch_normalization=batch_normalization, softmax=softmax)
        if print_cost and i % 100 == 0:
            print("cost iterate " + str(i) + " : " + str(J))

        # grads = backward_model(AL, Y, caches, regularation="DROPOUT")
        grads = backward_model(AL, Y, caches, regularation, batch_normalization=batch_normalization)

        if optimizer=="momentum":
            parameters, v = update_parameters_momentum(grads=grads, parameters=parameters, v=v, beta=beta, learning_rate=learning_rate, batch_normalization=batch_normalization)
        elif optimizer=="rmsprop":
            parameters, s = update_parameters_rmsprop(grads=grads, parameters=parameters, s=s, beta=beta, epsilon=epsilon, learning_rate=learning_rate, batch_normalization=batch_normalization)
        elif optimizer=="adam":
            parameters, v, s = update_parameters_adam(grads=grads, parameters=parameters, v=v, s=s, t=t, epsilon=epsilon, beta1=beta1, beta2=beta2, learning_rate=learning_rate, batch_normalization=batch_normalization)
        elif optimizer=="gd":
            parameters = update_parameters(grads, parameters, learning_rate, batch_normalization=batch_normalization)

        t = t + 1

    return parameters

def predict(X, y, parameters, batch_normalization=False, softmax=False):
    result, cache = forward_model(parameters, X, batch_normalization=batch_normalization)
    m = y.shape[1]
    pre = np.zeros((1, m))
    if softmax:
        pre = np.zeros((y.shape[0],m))

    for i in range(m):
        if softmax:
            pre[np.where(result[:, i] == np.max(result[:, i], axis=0))[0][0]][i] = 1
        else:
            if result[0, i] > 0.5:
                pre[0, i] = 1
            else:
                pre[0, i] = 0

    if softmax:
        accurancy = np.sum(pre * y) * 100 / result.shape[1]
    else:
        accurancy = ((np.dot(y, pre.T) + np.dot((1 - y), (1 - pre).T)) * 100) / result.shape[1]

    print("m: " + str(m))
    print("accurancy: " + str(accurancy))



train_x_orig, train_y, test_x_orig, test_y, classes = du.load_data()
# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

# train_y=np.tile(train_y,(2,1))
# train_y[1,:]=1-train_y[1,:]
#
# test_y=np.tile(test_y,(2,1))
# test_y[1,:]=1-test_y[1,:]

layers_dims = [12288, 20, 7, 5, 1]
parameters = L_layer_model(train_x, train_y, layers_dims, optimizer = "gd", regularation="L2", init_weight_func="ys", num_iterations = 300, batch_normalization=True, print_cost=True)
predict(train_x, train_y, parameters, batch_normalization=True,softmax=False)
predict(test_x, test_y, parameters, batch_normalization=True,softmax=False)

# ============ cost regularation begin ================
# def compute_cost_with_regularization_test_case():
#     np.random.seed(1)
#     Y_assess = np.array([[1, 1, 0, 1, 0]])
#     W1 = np.random.randn(2, 3)
#     b1 = np.random.randn(2, 1)
#     W2 = np.random.randn(3, 2)
#     b2 = np.random.randn(3, 1)
#     W3 = np.random.randn(1, 3)
#     b3 = np.random.randn(1, 1)
#     parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
#     a3 = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
#     return a3, Y_assess, parameters
#
# A3, Y_assess, parameters = compute_cost_with_regularization_test_case()
# print("cost = " + str(cost_function(A3, Y_assess, parameters, regularation="L2", lamd=0.1)))
# ============ cost regularation end ================

# ============ backward_propagation regularation begin ================
# def backward_propagation_with_regularization_test_case():
#     np.random.seed(1)
#     X_assess = np.random.randn(3, 5)
#     Y_assess = np.array([[1, 1, 0, 1, 0]])
#     cache = (np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
#          [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]]),
#   np.array([[ 0.        ,  3.32524635,  2.13994541,  2.60700654,  0.        ],
#          [ 0.        ,  4.1600994 ,  0.79051021,  1.46493512,  0.        ]]),
#   np.array([[-1.09989127, -0.17242821, -0.87785842],
#          [ 0.04221375,  0.58281521, -1.10061918]]),
#   np.array([[ 1.14472371],
#          [ 0.90159072]]),
#   np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
#          [-0.69166075, -3.47645987, -2.25194702, -2.65416996, -0.69166075],
#          [-0.39675353, -4.62285846, -2.61101729, -3.22874921, -0.39675353]]),
#   np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
#          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
#          [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]),
#   np.array([[ 0.50249434,  0.90085595],
#          [-0.68372786, -0.12289023],
#          [-0.93576943, -0.26788808]]),
#   np.array([[ 0.53035547],
#          [-0.69166075],
#          [-0.39675353]]),
#   np.array([[-0.3771104 , -4.10060224, -1.60539468, -2.18416951, -0.3771104 ]]),
#   np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]]),
#   np.array([[-0.6871727 , -0.84520564, -0.67124613]]),
#   np.array([[-0.0126646]]))
#     return X_assess, Y_assess, cache
#
# X_assess, Y_assess, cache2 = backward_propagation_with_regularization_test_case()
# caches = []
#
# (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache2
# linear_cache = (W1, X_assess, b1)
# activation_cache = (Z1, A1)
# cache = (linear_cache, activation_cache)
# caches.append(cache)
#
# linear_cache = (W2, A1, b2)
# activation_cache = (Z2, A2)
# cache = (linear_cache, activation_cache)
# caches.append(cache)
#
# linear_cache = (W3, A2, b3)
# activation_cache = (Z3, A3)
# cache = (linear_cache, activation_cache)
# caches.append(cache)
#
# grads = backward_model(A3, Y_assess, caches)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("dW3 = "+ str(grads["dW3"]))
# ============ backward_propagation regularation end ================

# ============ test drop out begin ===============
# def forward_propagation_with_dropout_test_case():
#     np.random.seed(1)
#     X_assess = np.random.randn(3, 5)
#     W1 = np.random.randn(2, 3)
#     b1 = np.random.randn(2, 1)
#     W2 = np.random.randn(3, 2)
#     b2 = np.random.randn(3, 1)
#     W3 = np.random.randn(1, 3)
#     b3 = np.random.randn(1, 1)
#     parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
#
#     return X_assess, parameters
#
#
# def backward_propagation_with_dropout_test_case():
#     np.random.seed(1)
#     X_assess = np.random.randn(3, 5)
#     Y_assess = np.array([[1, 1, 0, 1, 0]])
#     cache = (np.array([[-1.52855314, 3.32524635, 2.13994541, 2.60700654, -0.75942115],
#                        [-1.98043538, 4.1600994, 0.79051021, 1.46493512, -0.45506242]]),
#              np.array([[True, False, True, True, True],
#                        [True, True, True, True, False]], dtype=bool), np.array([[0., 0., 4.27989081, 5.21401307, 0.],
#                                                                                 [0., 8.32019881, 1.58102041, 2.92987024,
#                                                                                  0.]]),
#              np.array([[-1.09989127, -0.17242821, -0.87785842],
#                        [0.04221375, 0.58281521, -1.10061918]]), np.array([[1.14472371],
#                                                                           [0.90159072]]),
#              np.array([[0.53035547, 8.02565606, 4.10524802, 5.78975856, 0.53035547],
#                        [-0.69166075, -1.71413186, -3.81223329, -4.61667916, -0.69166075],
#                        [-0.39675353, -2.62563561, -4.82528105, -6.0607449, -0.39675353]]),
#              np.array([[True, False, True, False, True],
#                        [False, True, False, True, True],
#                        [False, False, True, False, False]], dtype=bool),
#              np.array([[1.06071093, 0., 8.21049603, 0., 1.06071093],
#                        [0., 0., 0., 0., 0.],
#                        [0., 0., 0., 0., 0.]]), np.array([[0.50249434, 0.90085595],
#                                                          [-0.68372786, -0.12289023],
#                                                          [-0.93576943, -0.26788808]]), np.array([[0.53035547],
#                                                                                                  [-0.69166075],
#                                                                                                  [-0.39675353]]),
#              np.array([[-0.7415562, -0.0126646, -5.65469333, -0.0126646, -0.7415562]]),
#              np.array([[0.32266394, 0.49683389, 0.00348883, 0.49683389, 0.32266394]]),
#              np.array([[-0.6871727, -0.84520564, -0.67124613]]), np.array([[-0.0126646]]))
#
#     return X_assess, Y_assess, cache
#
# # X_assess, parameters = forward_propagation_with_dropout_test_case()
# #
# # A3, caches = forward_model(parameters, X_assess, DROPOUT=True)
# # print ("A3 = " + str(A3))
# X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()
# (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
# caches = []
# linear_cache = [W1,X_assess,b1]
# activation_cache = [Z1,A1,D1]
# caches.append((linear_cache,activation_cache))
#
# linear_cache = [W2,A1,b2]
# activation_cache = [Z2,A2,D2]
# caches.append((linear_cache,activation_cache))
#
# linear_cache = [W3,A2,b3]
# D3 = np.ones((A3.shape[0], A3.shape[1]))
# activation_cache = [Z3,A3,D3]
# caches.append((linear_cache,activation_cache))
#
#
# gradients = backward_model(A3, Y_assess, caches, regularation="DROPOUT")
# print ("dA1 = " + str(gradients["dA1"]))
# print ("dA2 = " + str(gradients["dA2"]))

# ================ test dropout end ================

# ================= test optimizer =================
# def update_parameters_with_momentum_test_case():
#     np.random.seed(1)
#     W1 = np.random.randn(2, 3)
#     b1 = np.random.randn(2, 1)
#     W2 = np.random.randn(3, 3)
#     b2 = np.random.randn(3, 1)
#
#     dW1 = np.random.randn(2, 3)
#     db1 = np.random.randn(2, 1)
#     dW2 = np.random.randn(3, 3)
#     db2 = np.random.randn(3, 1)
#     parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
#     grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
#     v = {'dW1': np.array([[0., 0., 0.],
#                           [0., 0., 0.]]), 'dW2': np.array([[0., 0., 0.],
#                                                            [0., 0., 0.],
#                                                            [0., 0., 0.]]), 'db1': np.array([[0.],
#                                                                                             [0.]]),
#          'db2': np.array([[0.],
#                           [0.],
#                           [0.]])}
#     return parameters, grads, v
#
# def update_parameters_with_adam_test_case():
#     np.random.seed(1)
#     v, s = ({'dW1': np.array([[0., 0., 0.],
#                               [0., 0., 0.]]), 'dW2': np.array([[0., 0., 0.],
#                                                                [0., 0., 0.],
#                                                                [0., 0., 0.]]), 'db1': np.array([[0.],
#                                                                                                 [0.]]),
#              'db2': np.array([[0.],
#                               [0.],
#                               [0.]])}, {'dW1': np.array([[0., 0., 0.],
#                                                          [0., 0., 0.]]), 'dW2': np.array([[0., 0., 0.],
#                                                                                           [0., 0., 0.],
#                                                                                           [0., 0., 0.]]),
#                                         'db1': np.array([[0.],
#                                                          [0.]]), 'db2': np.array([[0.],
#                                                                                   [0.],
#                                                                                   [0.]])})
#     W1 = np.random.randn(2, 3)
#     b1 = np.random.randn(2, 1)
#     W2 = np.random.randn(3, 3)
#     b2 = np.random.randn(3, 1)
#
#     dW1 = np.random.randn(2, 3)
#     db1 = np.random.randn(2, 1)
#     dW2 = np.random.randn(3, 3)
#     db2 = np.random.randn(3, 1)
#
#     parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
#     grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
#
#     return parameters, grads, v, s
#
# parameters, grads, v = update_parameters_with_momentum_test_case()
#
# parameters, v = update_parameters_momentum(parameters=parameters, grads=grads, v=v, beta = 0.9, learning_rate = 0.01)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))
#
# parameters, grads, v, s = update_parameters_with_adam_test_case()
# parameters, v, s  = update_parameters_adam(parameters=parameters, grads=grads, v=v, s=s, t = 2)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))
# print("s[\"dW1\"] = " + str(s["dW1"]))
# print("s[\"db1\"] = " + str(s["db1"]))
# print("s[\"dW2\"] = " + str(s["dW2"]))
# print("s[\"db2\"] = " + str(s["db2"]))
# ============== end optimizer ======================

# ./  // 区别
# init参数除以 np.sqrt(layer_dims[l-1])
# 参数的更新：J是关于w的函数，需要计算哪个W的位置，能将dJ设置为0，
# 数据集的准备
# random随机数




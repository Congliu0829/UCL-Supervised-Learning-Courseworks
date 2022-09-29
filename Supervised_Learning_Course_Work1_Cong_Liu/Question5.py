# Library and parameter initialization
import random
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv
from numpy.linalg import norm
import pandas as pd

df = pd.read_csv("Boston-filtered.csv")
data = df.values

### function initialization for ex5
def kernel(x, t, sigma):
    '''
    return the gaussian kernel of two points, exp(-(||x-t||^2)/(2*sigma^2))

    Input: -x: array, any other points of input X

           -t: array, fixed test point

           -sigma: float, kernel parameter

    Output: float, kernel value
    '''
    return np.exp(-norm(x - t)**2 / (2*(sigma**2)))


def K_matrix(x, sigma):
    '''
    Get the kernel matrix for each point in X, size m*m
    Using bradcase to speed up

    Input: -x: array, input training data

           -sigma: float, kernel parameter

    Output: array (m*m), with each element representing kernel value
    '''
    X_wide = np.reshape(np.tile(x, len(x)), (len(x),len(x),len(x[0]))) #broadcast X widely
    X_deep = X_wide.transpose(1,0,2)
    return np.exp(-np.sum((X_wide - X_deep)**2, axis=2)/(2*sigma**2))


def alpha(x, y, sigma, lam):
    '''
    Get the alpha vector with regulization parameter and kernels

    Input: -x: array, input training data

           -y: array, input training labels

           -sigma: float, kernel parameter

           -lam: ragulator parameter

    Output: array (m*1), alpha vector for
    '''
    K = K_matrix(x, sigma)
    return np.matmul(np.linalg.inv(K + len(x) * lam * np.identity(len(x))), y)
    
    
def pair(sigma_vector, lam_vector):
    '''
    Pair each sigma and lambda value from parameter list

    Input: -sigma_vector: array, sigma list

           -lam_vector: array, lambda list

    Output: list, each element is a pair of sigma and lambda values
    '''
    lst = [] 
    for sigma in sigma_vector:
        for lam in lam_vector:
            lst.append([sigma, lam])
    return lst


def predict(x_train, x_test, alpha_vector, sigma, lam, train=True):
    '''
    Get predicted y with different pair of sigma and lambda values

    #####Using broadcast to speed up######

    Input: -x_train: array, input training data

           -x_test: array, input testing data

           -alpha_vector: array, calculated alpha vector

           -sigma: float, kernel parameter

           -lam: float, ragulator parameter

           -train: bool, True for training prediction, False for testing prediction

    Output: array with predicted y array
    '''
    x = x_train if train else x_test
    x_test_tile = np.reshape(np.tile(x,len(x_train)), (len(x),len(x_train),len(x[0])))#broadcast
    x_train_tile = np.reshape(np.tile(x_train,len(x)), (len(x_train),len(x),len(x_train[0])))#broadcast
    x_train_tile = x_train_tile.transpose(1,0,2)
    alpha_vector_tile = np.reshape(np.tile(alpha_vector,len(x)), (len(x),len(alpha_vector)))#broadcast
    return np.sum(alpha_vector_tile * np.exp(-np.sum((x_test_tile - x_train_tile)**2, axis=2)/(2*sigma**2)), axis = 1)
    

def cal_mse(y, y_pred):
    '''
    Compute mse with given y data and predicted y data

    Input: -y: array, y data

           -y_pred: array, predicted y data

    Output: int, the value of mse
    '''
    mse = 0
    for i in range(len(y)):
        mse += (y[i] - y_pred[i]) ** 2
        # mse += abs(y[i] - y_pred[i])
    return mse / len(y)

def cross_val(x, y, sigma_vector, lam_vector):
    '''
    Five-fold cross validtion method, return the mse_list with all results

    Input: -x: array, input training data

           -y: array, input training labels

           -sigma_vector: list, list of sigma values

           -lam_vector: list, list of lambda values

    Output: list of all mse scores
    '''
    ### five fold initializtion
    size = len(y)
    fold_size = size//5
    x_fold_list = []
    y_fold_list = [] 
    ### Divide the training list into 5 roughly equal subsets
    for i in range(5):
        if i < 4:
            x_fold_list.append(x[fold_size*i:fold_size*(i+1)])
            y_fold_list.append(y[fold_size*i:fold_size*(i+1)])
        else:
            x_fold_list.append(x[fold_size*i:])
            y_fold_list.append(y[fold_size*i:])

    ### In each iteration, one of the 5 subsets will be the validation set
    sl_list = pair(sigma_vector, lam_vector)
    cv_error_list = np.zeros(len(sl_list)) #13*15 pairs of parameter
    for i in range(5):    
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        idx_list = [0,1,2,3,4]
        idx_list.remove(i)
        X_test.extend(x_fold_list[i])
        y_test.extend(y_fold_list[i])
        for idx in idx_list:
            X_train.extend(x_fold_list[idx])
            y_train.extend(y_fold_list[idx])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Train data and validate the performance for 13*15*5 times
        mse_list = []
        for vector_pair in sl_list:
            sigma, lam = vector_pair[0], vector_pair[1]
            alpha_vector = alpha(X_train, y_train, sigma, lam)
            y_pred = predict(X_train, X_test, alpha_vector, sigma, lam, train=False)
            mse = cal_mse(y_test, y_pred)
            mse_list.append(mse) #13*15
        cv_error_list += np.array(mse_list)
    cv_error_list /= 5
    return cv_error_list


lam_vector = [2**(-40+i) for i in range(15)]
sigma_vector = [2**7 * 2**(0.5*i) for i in range(13)]

#### 5(a) #########################################################################################################################################################################
# Split the dataset to train and test, then get the index of minimal error using cross validation
np.random.shuffle(data)
data_x = data[:, :-1]
data_y = data[:, -1]
test_size = len(data) // 3
X_test = data_x[:test_size]
y_test = data_y[:test_size]
X_train = data_x[test_size:]
y_train = data_y[test_size:]
cv_error_list = np.array(cross_val(X_train, y_train, sigma_vector, lam_vector))
min_idx = np.argmin(cv_error_list)

# The index of row and column are the optimal choice of sigma and lambda respectively
if min_idx != 0:
    sigma_idx = min_idx // 13 if min_idx % 13 != 0 else min_idx // 13 - 1
    lam_idx = min_idx % 15 if min_idx % 15 != 0 else min_idx % 15 - 1
    print("optimal log2(sigma_factor): {0: 3f}, optimal log2(lambda_factor): {1: 3f}".format(np.log2(sigma_vector[sigma_idx]), np.log2(lam_vector[lam_idx])))
else:
    print("optimal log2(sigma_factor): {0: 3f}, optimal log2(lambda_factor): {1: 3f}".format(np.log2(sigma_vector[sigma_idx]), np.log2(lam_vector[lam_idx])))

o_sigma, o_lam = sigma_vector[sigma_idx], lam_vector[lam_idx]

### 5-(b) #########################################################################################################################################################################
# Reshape x and y in order to plot the 3-d figure
sigma_vector_tile = np.tile(np.array(np.log2(sigma_vector)), len(lam_vector))
x_axis = np.reshape(sigma_vector_tile, (len(lam_vector), len(sigma_vector))).transpose(1,0).flatten()
lam_vector_tile = np.tile(np.array(np.log2(lam_vector)), len(sigma_vector))
y_axis = np.reshape(lam_vector_tile, (len(sigma_vector), len(lam_vector))).flatten()

# Plot the figure
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax = Axes3D(fig)
ax.scatter(x_axis, y_axis, cv_error_list)
ax.set_xlabel('$log_2(\sigma)$', fontsize=12, rotation=150)
ax.set_ylabel('$log_2(\gamma)$', fontsize=12, rotation=150)
ax.set_zlabel('cross validation error', fontsize=9)
plt.savefig('1-5-b', dpi=500)
plt.show()

#### 5-(c) #########################################################################################################################################################################
# Using function defined before to calculate training and testing mse
sigma, lam = o_sigma, o_lam
alpha_vector = alpha(X_train, y_train, sigma, lam)
y_train_pred = predict(X_train, X_test, alpha_vector, sigma, lam, train=True)
y_test_pred = predict(X_train, X_test, alpha_vector, sigma, lam, train=False)
mse_train = cal_mse(y_train, y_train_pred)
mse_test = cal_mse(y_test, y_test_pred)

print("5-(c) mse_train: ", mse_train)
print("5-(c) mse_test: ", mse_test)

#### 5-(d) #########################################################################################################################################################################
from Question4 import forward_train
from Question4 import forward_test
# Re-initialize data for 20 runs
df = pd.read_csv("Boston-filtered.csv")
data = df.values
training_x = data[:, :-1]
training_y = data[:, -1]

ex4_naive_train = []
ex4_naive_test = []
ex4_1_train = []
ex4_1_test = []
ex4_12_train = []
ex4_12_test = []

ex5_train = []
ex5_test = []

# Run 20 epochs
for epoch in range(20):
    np.random.shuffle(data)
    data_x = data[:, :-1]
    data_y = data[:, -1]
    test_size = len(data) // 3
    x_test = data_x[:test_size]
    y_test = data_y[:test_size]
    x_train = data_x[test_size:]
    y_train = data_y[test_size:]
    
    #### EX4 a,c,d
    ### naive regression// num_feat = 0
    w_naive, mse_naive_train = forward_train(x_train, y_train, 0)
    mse_naive_test = forward_test(x_test, y_test, w_naive, 0)
    ex4_naive_train.append(mse_naive_train)
    ex4_naive_test.append(mse_naive_test)

    ### 1 attribute regression
    w_1, mse_1_train = forward_train(x_train, y_train, 1)
    mse_1_test = forward_test(x_test, y_test, w_1, 1)
    ex4_1_train.append(mse_1_train)
    ex4_1_test.append(mse_1_test)
    ### 12 attributes
    w_12, mse_12_train = forward_train(x_train, y_train, 12)
    mse_12_test = forward_test(x_test, y_test, w_12, 12)
    ex4_12_train.append(mse_12_train)
    ex4_12_test.append(mse_12_test)


    #### EX5 a,c
    X_train = x_train
    X_test = x_test
    cv_error_list = np.array(cross_val(X_train, y_train, sigma_vector, lam_vector))
    min_idx = np.argmin(cv_error_list)
    if min_idx != 0:
        sigma_idx = min_idx // 13 if min_idx % 13 != 0 else min_idx // 13 - 1
        lam_idx = min_idx % 15 if min_idx % 15 != 0 else min_idx % 15 - 1
        print("optimal log2(sigma_factor): {0: 6f}, optimal log2(lambda_factor): {1: 9f}".format(np.log2(sigma_vector[sigma_idx]), np.log2(lam_vector[lam_idx])))
    else:
        print("optimal log2(sigma_factor): {0: 6f}, optimal log2(lambda_factor): {1: 9f}".format(sigma_vector[sigma_idx], lam_vector[lam_idx]))
    o_sigma, o_lam = sigma_vector[sigma_idx], lam_vector[lam_idx]

    sigma, lam = o_sigma, o_lam
    alpha_vector = alpha(X_train, y_train, sigma, lam)
    y_train_pred = predict(X_train, X_test, alpha_vector, sigma, lam, train=True)
    y_test_pred = predict(X_train, X_test, alpha_vector, sigma, lam, train=False)
    mse_train = cal_mse(y_train, y_train_pred)
    mse_test = cal_mse(y_test, y_test_pred)
    ex5_train.append(mse_train)
    ex5_test.append(mse_test)


print("train MSE of EX4 naive: {0:.2f} ± {1:.2f}".format(np.mean(ex4_naive_train), np.sqrt(np.var(ex4_naive_train))))

print("test MSE of EX4 naive: {0:.2f} ± {1:.2f}".format(np.mean(ex4_naive_test), np.sqrt(np.var(ex4_naive_test))))

print("train MSE of EX4 with one attribute:")
print(np.mean(ex4_1_train, axis=0), np.sqrt(np.var(ex4_1_train, axis=0)))

print("test MSE of EX4 with one attribute:")
print(np.mean(ex4_1_test, axis=0), np.sqrt(np.var(ex4_1_test, axis=0)))

print("train MSE of EX4 with all attributes: {0:.2f} ± {1:.2f}".format(np.mean(ex4_12_train), np.sqrt(np.var(ex4_12_train))))

print("test MSE of EX4 with all attributes: {0:.2f} ± {1:.2f}".format(np.mean(ex4_12_test), np.sqrt(np.var(ex4_12_test))))

print("train MSE of EX5 Kernel: {0:.2f} ± {1:.2f}".format(np.mean(ex5_train), np.sqrt(np.var(ex5_train))))

print("test MSE of EX5 Kernel: {0:.2f} ± {1:.2f}".format(np.mean(ex5_test), np.sqrt(np.var(ex5_test))))
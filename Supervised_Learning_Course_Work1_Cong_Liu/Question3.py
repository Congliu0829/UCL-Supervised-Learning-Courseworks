# Initialization for librarys and k lists
import matplotlib.pyplot as plt
import numpy as np
k_list = [i for i in range(1, 19)]

# Define functions for all the rest of exercise 3
from numpy.linalg import inv

def gen_data(x, y, k):
    '''
    For a given input and degree k, return its feature map with 
        k-sin-function basis, and y data

    Input: -x: array, input training data

           -y: array, input training labels

           -k: int, degree of the basis

    Output: - array but change each element in the original x to an
              array with k elements 

            - y data
    '''
    feature_mapped_x = [[] for i in range(len(x))] # x with dimention (m,1) to dimention (m,n)
    for i in range(len(x)):
        feature_mapped_x[i] = [np.sin(np.pi * x[i] * m) for m in range(1, k)]
        feature_mapped_x[i].append(1) #bias term in X

    return np.array(feature_mapped_x), y

def compute_weight(X, y):   
    '''
    Compute w = (X_T·X)^(-1)X_T·y

    Input: -X: array, feature map of x

           -y: array, y data

    Output: array with dimention of features
    '''
    return np.dot(np.matmul(inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)

def predict_y(w, X):
    '''
    Get predicted y data with y=w^T·X
    
    Input: -X: array, training input

           -w: array, trained weight

    Output: -y_hat: array, predicted y with training data
    '''
    return np.matmul(X, w)

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
    return mse / len(y)

def forward_train(x, y, k):
    '''
    Get trained weight and calculate trained mse

    Input: -y: array, y data

           -x: array, input training data

           -k: int, sin-function parameter

    Output: -w: trained weight with dimention n

            -mse: int, the value of training mse
    '''
    X, y = gen_data(x, y, k)
    w = compute_weight(X, y)
    y_pred = predict_y(w, X)
    mse = cal_mse(y_pred, y)
    return w, mse

def forward_test(x, y, w, k):
    '''
    Get test mse with test data and test labels

    Input: -x: array, testing input

           -k: int, sin-function parameter

           -w: array, trained weight

    Output: int, value of test mse
    '''
    X, y = gen_data(x, y, k)
    y_pred = predict_y(w, X)
    mse = cal_mse(y_pred, y)
    return mse

def plotcurve(x, y, x_pred, y_pred, mse, k):
    '''
    Plot the trained curve with weight

    Input: -x: array, training input

           -y: array, training label

           -x_pred: array, generated x points waiting to be predicted

           -y_pred: array, predicted test y

           -k: int, sin-function parameter
    '''
    fig, ax = plt.subplots()
    ax.scatter(x, y, label='data', c  = 'r')
    ax.plot(x_pred, y_pred,  label='predicted function', c = "b")
    plt.title('k = {0}, MSE = {1:.4f}'.format(k, mse))
    plt.legend()
    plt.grid()
    plt.savefig("1-2-b")
    plt.show()

##### 3- original data plot #########################################################################################################################################################################
# Data-point initialization
x = np.random.uniform(0, 1, 50)
epsilon = np.random.normal(0, 0.07, 50)
g = (np.sin(2 * np.pi * x))**2 + epsilon

# Print data points and functions
fig, ax = plt.subplots()
ax.scatter(x, g)
# x.sort()
function_x = np.arange(0, 1, 0.0001)
ax.plot(function_x, (np.sin(2 * np.pi * function_x))**2)
plt.grid()
plt.show()


#### 3-(b) #########################################################################################################################################################################
mse_list = []
w_list = []
for k in k_list:
    # For different k, get different feature maps
    X, y = gen_data(x, g, k)
    w = compute_weight(X, y)
    w_list.append(w)
    y_pred = predict_y(w, X)
    mse = cal_mse(y, y_pred)
    mse_list.append(mse)
    x_pred = np.arange(0,1,0.0001)

# Plot trained mse
plt.plot(k_list, np.log(mse_list), label='train')
plt.legend()
plt.xlabel("Polynomial dimensions")
plt.ylabel("ln(Training Mean Squared Error)")
plt.grid()
plt.savefig("1-3-b", dpi=500)
plt.show()


#### 3-(c) #########################################################################################################################################################################
# Generate 1000 testing points
x_test = np.random.uniform(0, 1, 1000)
epsilon_test = np.random.normal(0, 0.07, 1000)
g_test = (np.sin(2 * np.pi * x_test))**2 + epsilon_test

# Use trained w to get test mse
mse_test_list = []
for k in k_list:
    X_test, y_test = gen_data(x_test, g_test, k)
    y_test_pred = predict_y(w_list[k-1], X_test)
    mse_test = cal_mse(y_test, y_test_pred)
    mse_test_list.append(mse_test)

# Plot the test mse curve
plt.plot(k_list, np.log(mse_test_list), label='test')
plt.legend()
plt.xlabel("Polynomial dimensions")
plt.ylabel("ln(Testing Mean Squared Error)")
plt.grid()
plt.savefig("1-3-c", dpi=500)
plt.show()


#### 3-(d) #########################################################################################################################################################################
mse_train_list = []
mse_test_list = []
for k in k_list:
    mse_train = 0
    mse_test = 0
    epoch = 0
    while epoch < 100:
        # Generate data for each epoch
        x = np.random.uniform(0, 1, 50)
        epsilon = np.random.normal(0, 0.07, 50)
        g = (np.sin(2 * np.pi * x))**2 + epsilon
        x_test = np.random.uniform(0, 1, 1000)
        epsilon_test = np.random.normal(0, 0.07, 1000)
        g_test = (np.sin(2 * np.pi * x_test))**2 + epsilon_test
        ### train and get trained mse
        w, mse_t = forward_train(x, g, k)
        mse_train += mse_t
        ### test to get test mse
        mse_test += forward_test(x_test, g_test, w, k)
        epoch += 1
    # Get the average mse values
    mse_train_list.append(mse_train/100)
    mse_test_list.append(mse_test/100)

# Plot curves with ln(avg)
plt.plot(k_list, np.log(mse_train_list), label='train')
plt.plot(k_list, np.log(mse_test_list), label='test')
plt.legend()
plt.xlabel("Polynomial dimensions")
plt.ylabel("ln(Mean Squared Error)")
plt.grid()
plt.savefig("1-3-d", dpi=500)
plt.show()



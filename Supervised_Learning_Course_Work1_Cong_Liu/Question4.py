# Library initialization and Get the dataset
from numpy.linalg import inv
import pandas as pd
import numpy as np


# Define functions for all the rest of exercise 4
def gen_data(x, y, num_feat):
    '''
    For a given input and the number of features, return pre-processed data

    Input: -x: array, input training data

           -y: array, input training labels

           -num_feat: int, number of features in [0,1,12], 0 represents naive regression

    Output: - array of training data 

            - y data
    '''
    if num_feat == 0:
        size = len(y)
        X = np.ones((size, 1))
        return np.array(X), y
    if num_feat == 1:
        X_list = []
        for i in range(12):
            X = [[] for _ in range(y.size)]
            for idx in range(y.size):
                X[idx].append(x[idx][i])
                X[idx].append(1) #bias
            X_list.append(np.array(X))
        return X_list, y
    if num_feat == 12:
        X = []
        for line in x:
            a = np.append(line, 1)
            X.append(a)
        return np.array(X), y

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

def forward_train(x, y, num_feat):
    '''
    Get trained weight and calculate trained mse

    Input: -y: array, y data

           -x: array, input training data

           -num_feat: int, number of features

    Output: -w_list: array, trained weights with dimention n and features

            -mse_list: array, the array of training mse's
    '''
    if num_feat == 1:
        mse_list = []
        w_list = []
        X_list, y = gen_data(x, y, num_feat)
        for idx in range(len(X_list)):
            w = compute_weight(X_list[idx], y)
            w_list.append(w)
            y_pred = predict_y(w, X_list[idx])
            mse = cal_mse(y_pred, y)
            mse_list.append(mse)
        return w_list, np.array(mse_list)

    else:
        X, y = gen_data(x, y, num_feat)
        w = compute_weight(X, y)
        y_pred = predict_y(w, X)
        mse = cal_mse(y_pred, y)
        return w, mse

def forward_test(x, y, w, num_feat):
    '''
    Get test mse with test data and test labels

    Input: -x: array, testing input

           -y: array, testing labels

           -w: array, trained weight

           -num_feat: int, number of attributes

    Output: int/array, value of test mse
    '''
    if num_feat == 1:
        mse_list = []
        X_list, y = gen_data(x, y, num_feat)
        for idx in range(len(X_list)):
            y_pred = predict_y(w[idx], X_list[idx])
            mse = cal_mse(y_pred, y)
            mse_list.append(mse)
        return np.array(mse_list)
    else:
        X, y = gen_data(x, y, num_feat)
        y_pred = predict_y(w, X)
        mse = cal_mse(y_pred, y)
        return mse

def plotcurve(x, y, y_predict, mse):
    '''
    Plot the trained curve with weight

    Input: -x: array, training input

           -y: array, training label

           -x_pred: array, generated x points waiting to be predicted

           -y_pred: array, predicted test y

           -k: int, sin-function parameter
    '''
    fig, ax = plt.subplots()
    ax.scatter(x, y, label='data')
    ax.scatter(x, y_pred, label='predict')
    plt.title('k = {0}, MSE = {1:.4f}'.format(k, mse))
    plt.legend()
    plt.grid()
    plt.show()

# Pre-process data and get training data and labels
df = pd.read_csv("Boston-filtered.csv")
data = df.values

mse_naive_train_avg = 0
mse_naive_test_avg = 0
mse_1_train_avg_list = np.zeros(12)
mse_1_test_avg_list = np.zeros(12)
mse_12_train_avg = 0
mse_12_test_avg = 0

# Run 20 epochs
for epoch in range(20):
    # Shuffle data and get train/test data with 1/3 split
    np.random.shuffle(data)
    data_x = data[:, :-1]
    data_y = data[:, -1]
    test_size = len(data) // 3
    x_test = data_x[:test_size]
    y_test = data_y[:test_size]
    x_train = data_x[test_size:]
    y_train = data_y[test:]
    ### naive regression// num_feat = 0
    w_naive, mse_naive_train = forward_train(x_train, y_train, 0)
    mse_naive_train_avg += mse_naive_train
    mse_naive_test = forward_test(x_test, y_test, w_naive, 0)
    mse_naive_test_avg += mse_naive_test
    ### 1 attribute regression
    w_1_list, mse_1_train_list = forward_train(x_train, y_train, 1)
    mse_1_train_avg_list += mse_1_train_list
    mse_1_test_list = forward_test(x_test, y_test, w_1_list, 1)
    mse_1_test_avg_list += mse_1_test_list
    ### 12 attributes
    w_12, mse_12_train = forward_train(x_train, y_train, 12)
    mse_12_train_avg += mse_12_train
    mse_12_test = forward_test(x_test, y_test, w_12, 12)
    mse_12_test_avg += mse_12_test

mse_naive_train_avg /= 20
mse_naive_test_avg /= 20
mse_1_train_avg_list /= 20
mse_1_test_avg_list /= 20
mse_12_train_avg /= 20
mse_12_test_avg /= 20

# 4-(a) Naive regression training mse (mean value) ###########################################
print("Naive regression training mse", mse_naive_train_avg)

# 4-(a) Naive regression testing mse (mean value) ###########################################
print("Naive regression testing mse", mse_naive_test_avg)

# 4-(c) training mse with single attribute ###########################################
print("training mse with single attribute", mse_1_train_avg_list) 

# 4-(c) testing mse with single attribute ###########################################
print("testing mse with single attribute", mse_1_test_avg_list)

# 4-(d) training mse with 12 attributes together ###########################################
print("training mse with 12 attributes together", mse_12_train_avg)

# 4-(d) testing mse with 12 attributes together ###########################################
print("testing mse with 12 attributes together", mse_12_test_avg)




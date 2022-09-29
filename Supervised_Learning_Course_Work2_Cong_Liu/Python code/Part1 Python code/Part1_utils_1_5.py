# Define functions for PART1.5
from numpy.linalg import norm
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt 

def random_split(X, y, split_ratio=1/5):
    # This function aims to randomly split data to 80% for training, 20% for testing
    '''
        Input: 
            -X whole data matrix m*n, m represents data points, n represents data dimensions
            -y whole data label vector m*1
        
        Output:
            -X_train training data matrix
            -X_test testing data matrix
            -y_train training data label
            -y_test testing data label 
    '''
    data_idx = np.array([i for i in range(len(X))])
    np.random.shuffle(data_idx)
    test_size = int(len(X) * split_ratio)
    X_test = np.array([X[data_idx[i]] for i in range(test_size)])
    y_test = np.array([y[data_idx[i]] for i in range(test_size)])
    X_train = np.array([X[data_idx[i]] for i in range(test_size, len(X))])
    y_train = np.array([y[data_idx[i]] for i in range(test_size, len(y))])
    return X_train, y_train, X_test, y_test


def cross_validation(X, y, k):
    # This function aims to provide two list, each of them containing 
    # 5 sets of training data/label
    '''
        Input: 
            -X: whole data matrix m*n, m represents data points, n represents data dimensions
            -y: whole data label vector m*1

        Output:
            -X_train: list, containing 5 sets of training data
            -y_train: list, containing 5 sets of training label
    '''
    data_idx = [i for i in range(len(X))]
    np.random.shuffle(data_idx)
    len_train = len(X)//k
    X_train = [X[data_idx[i*len_train:(i+1)*len_train]] for i in range(k-1)]
    X_valid = X[data_idx[(k-1)*len_train:]]
    X_train.append(X_valid)

    y_train = [y[data_idx[i*len_train:(i+1)*len_train]] for i in range(k-1)]
    y_valid = y[data_idx[(k-1)*len_train:]]
    y_train.append(y_valid)

    return X_train, y_train


def kernel(p, q, c):
    return np.exp(-c* norm(p-q)**2)


def K_matrix(X1, X2, c):
    '''
        Get the kernel matrix for each point in X, size m*m
        Using bradcase to speed up

        Input: -x: array, input training data
            -sigma: float, kernel parameter
        Output: array (m*m), with each element representing kernel value
    '''
    out = np.zeros((len(X2),len(X1)))
    for i, x in enumerate(X2):
        out[i] = np.exp(-c*np.sum((x - X1)**2, axis=1))
    return out       


def sign(x):
    #This function realize the sign function
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0


def train(X_train, y_train, alpha, num_class, kernel_mat, c):
    '''
        Input:
            -X_train, training data matrix with size m_train * n
            -y_train, training data label vector with size m_train * 1
            -alpha, weight vector with size m_train * num_class
            -num_class, int, number of class, in our case is 10
            -kernel_mat, kernel matrix of training set.
            -c, parameter inside kernel function
        
        Output:
            -mistakes, int, number of mistakes detected in training one epoch
            -end-start, float, time spent on training one epoch
            -alpha, updated weighting vector, m_train * num_class
    '''
    start = time.time()
    mistakes = 0
    for i in range(len(X_train)):
        max_val = -float('inf')
        max_idx = -1
        for j in range(num_class):
            y_for_now = 1 if y_train[i] == j else -1
            pred_val = np.dot(kernel_mat[i], alpha[j])
            if sign(pred_val) != y_for_now: #predict wrong value; alpha[t] = y[t]
                if sign(pred_val) == 0:
                    alpha[j][i] = y_for_now
                alpha[j][i] -= sign(pred_val) #update alpha

            else: #predict right value, record confidence
                if pred_val > max_val:
                    max_val = pred_val #assign new max value
                    max_idx = j #record idx with max confidence
        #wrong prediction
        if max_idx != y_train[i]:
            mistakes += 1
    end = time.time()
        
    return mistakes, end-start, alpha

def test(X_train, X_test, y_test, alpha, num_class, kernel_mat, c):
    '''
        Input:
            -X_train, training data matrix with size m_train * n
            -y_train, training data label vector with size m_train * 1
            -alpha, weight vector with size m_train * num_class
            -num_class, int, number of class, in our case is 10
            -kernel_mat, kernel matrix of testing set.
            -c, parameter inside kernel function
        Output:   
            -mistakes, int, number of mistakes detected in testing one epoch
            -end-start, float, time spent on testing one epoch
    '''
    import time
    start = time.time()
    mistakes = 0
    for i in range(len(X_test)):
        max_val = -float('inf')
        for j in range(num_class):
            y_for_now = 1 if y_test[i] == j else -1
            pred_val = np.dot(kernel_mat[i], alpha[j])
            if pred_val > max_val:
                max_val = pred_val
                max_idx = j
        #wrong prediction        
        if max_idx != y_test[i]:
            mistakes += 1
    end = time.time()
    return mistakes, end-start


def test_confusion(X_train, X_test, y_test, alpha, num_class, kernel_mat, d, confusion_matrix):
    #This function aims to provide a test containing confusion matrix
    '''
        Input:
            -X_train, training data matrix with size m_train * n
            -y_train, training data label vector with size m_train * 1
            -alpha, weight vector with size m_train * num_class
            -num_class, int, number of class, in our case is 10
            -d, parameter inside kernel function
            -confusion_matrix, matrix filled with all 0s with size 10*10
                each entry ij represents the error rate of label i 
                being wrongly predict to label j
        Output:
            -mistakes, int, number of mistakes detected in testing one epoch
            -end-start, float, time spent on testing one epoch
            -confusion_matrix/len(X_test) updated normalized confusion matrix
    '''
    import time
    start = time.time()
    mistakes = 0
    for i in range(len(X_test)):
        max_val = -float('inf')
        for j in range(num_class):
            y_for_now = 1 if y_test[i] == j else -1
            pred_val = np.dot(kernel_mat[i], alpha[j])
            # pred_val = pred(X_train, X_test[i], alpha[j], d)
            if pred_val > max_val:
                max_val = pred_val
                max_idx = j
        #wrong prediction        
        if max_idx != y_test[i]:
            mistakes += 1
            confusion_matrix[y_test[i]][max_idx] += 1
    end = time.time()
    return mistakes, end-start, confusion_matrix/len(X_test)



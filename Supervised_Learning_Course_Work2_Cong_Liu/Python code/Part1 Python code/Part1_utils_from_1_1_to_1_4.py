import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt 

# Define functions for PART1
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



def kernel(x1, x2, d):
    # This function represents a kernel function
    return (np.dot(x1, x2))**d

def sign(x):
    # This function represents a sign function
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

def train(X_train, y_train, alpha, num_class, d):
    # This function realize the training process of one epoch of perceptron with polynomial kernel
    '''
        Input:
            -X_train, training data matrix with size m_train * n
            -y_train, training data label vector with size m_train * 1
            -alpha, weight vector with size m_train * num_class
            -num_class, int, number of class, in our case is 10
            -d, polynomial degree in kernel
        
        Output:
            -mistakes, int, number of mistakes detected in training one epoch
            -end-start, float, time spent on training one epoch
            -alpha, updated weighting vector, m_train * num_class
    '''
    start = time.time()
    mistakes = 0
    kernel_mat = kernel(X_train, np.transpose(X_train), d) 
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


def test(X_train, X_test, y_test, alpha, num_class, d, record_box=None, record=False):
    # This function aims to realize the testing process of trained perceptron with polynomial kernel
    '''
        Input:
            -X_train, training data matrix with size m_train * n
            -y_train, training data label vector with size m_train * 1
            -alpha, weight vector with size m_train * num_class
            -num_class, int, number of class, in our case is 10
            -d, polynomial degree in kernel
            -record_box, optional input, a record_box with size m_train * 2, 
               record[i][0], represents mistake number of i-th instance 
               record[i][1], represents whole mistake confidence of i-th instance
            -record, bool, record mistakes if True
        Output:   
            -mistakes, int, number of mistakes detected in testing one epoch
            -end-start, float, time spent on testing one epoch
            -record_box, updated record_box
    '''
    if not record:
        start = time.time()
        mistakes = 0
        kernel_mat = kernel(X_test, np.transpose(X_train), d)
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
    else:
        start = time.time()
        mistakes = 0
        kernel_mat = kernel(X_test, np.transpose(X_train), d)
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
                record_box[i][0] += 1
                record_box[i][1] += abs(max_val)
                mistakes += 1
        end = time.time()
        return mistakes, end-start, record_box


def test_confusion(X_train, X_test, y_test, alpha, num_class, d, confusion_matrix):
    '''
        Input:
            -X_train, training data matrix with size m_train * n
            -y_train, training data label vector with size m_train * 1
            -alpha, weight vector with size m_train * num_class
            -num_class, int, number of class, in our case is 10
            -d, polynomial degree in kernel
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
    kernel_mat = kernel(X_test, np.transpose(X_train), d)
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
            confusion_matrix[y_test[i]][max_idx] += 1
    end = time.time()
    return mistakes, end-start, confusion_matrix/len(X_test)
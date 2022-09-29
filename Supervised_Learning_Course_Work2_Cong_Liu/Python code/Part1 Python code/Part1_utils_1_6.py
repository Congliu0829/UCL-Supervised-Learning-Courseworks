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


def cal_mat(num_class):
    # This function aims to create a manipulation matrix used to convert 
    # the 45 outputs from 45 classifers to voting results for 10 labels
    '''
        Input: 
            -num_class: int, number of class
        Output:
            -cal_mat: manipulation matrix, used to convert the 45 outputs to voting results for 10 labels
    '''
    cal_mat = np.zeros((num_class, int(num_class*(num_class-1)/2)))
    #add 1
    start_idx = 0
    for p in range(len(cal_mat)):
        cal_mat[p][start_idx: start_idx + num_class-p-1] = 1
        start_idx = start_idx+num_class-p-1
    #add -1
    #row 1
    for i in range(1, num_class):
        cal_mat[i][i-1] = -1
    #row 2
    for i in range(2, num_class):
        cal_mat[i][i+7] = -1
    #row 3
    for i in range(3, num_class):
        cal_mat[i][i+14] = -1
    #row 4
    for i in range(4, num_class):
        cal_mat[i][i+20] = -1
    #row 5
    for i in range(5, num_class):
        cal_mat[i][i+25] = -1
    #row 6
    for i in range(6, num_class):
        cal_mat[i][i+29] = -1
    #row 7
    for i in range(7, num_class):
        cal_mat[i][i+32] = -1
    #row 8
    for i in range(8, num_class):
        cal_mat[i][i+34] = -1
    #row 9
    cal_mat[-1][-1] = -1
    return cal_mat


def kernel(x1, x2, d):
    # Polynomial kernel
    return (np.dot(x1, x2))**d

def sign(x):
    # Sign function
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

def get_idx(label):
    # This function aims to provide the index list of classifier corresponding to the input label
    '''
        Input: 
            -label: int, label range from 0 - 9
        Output:
            -idx_list 1: the index list of classifier corresponding to the main target label
            -idx_list 2: the index list of classifier corresponding to the side target label
            By main and side, i.e. classifer 01 tends to classify label 0 out of label 1
            The main target label is label 0,
            while the side target label is label 1.
    '''
    idx_list1 = [int(((9-i+1)+9)*i/2+label-1-i) for i in range(label)]
    idx_list2 = [int(((9-label+1)+9)*label/2 + i) for i in range(10-label-1)]
    return idx_list2, idx_list1


def train(X_train, y_train, alpha, num_class, kernel_mat, d):
    # This function realize the training process of one epoch of perceptron with polynomial kernel (OvO)
    '''
        Input:
            -X_train, training data matrix with size m_train * n
            -y_train, training data label vector with size m_train * 1
            -alpha, weight vector with size m_train * num_class
            -num_class, int, number of class, in our case is 10
            -kernel_mat, training kernel matrix
            -d, polynomial degree in kernel
        
        Output:
            -mistakes, int, number of mistakes detected in training one epoch
            -end-start, float, time spent on training one epoch
            -alpha, updated weighting vector, m_train * num_class
    '''
    start = time.time()
    mistakes = 0
    calculate_mat = cal_mat(10)
    for i in range(len(X_train)):
        #alpha in 45*len(X_train)
        res = np.dot(alpha, kernel_mat[i]) #45*1
        if res.tolist() == [0.0 for _ in range(int(num_class * (num_class-1) * 1/2))]:
            mistakes += 1
        res = np.array([1 if i > 0 else -1 for i in res]) #45*1
        res = np.dot(calculate_mat, res) #10*1 
        vote_res = np.argmax(res)
        if vote_res != y_train[i]:
            mistakes += 1

        #For the specific y_train[i], we figure out which classifier should be updated.
        #Obtain corresponding index list
        forward_idx, backward_idx = get_idx(y_train[i])
        #In list of main target classifer, we transfer y_train[i] to 1
        if forward_idx != []:
            for idx in forward_idx:
                y_for_now = 1
                pred_val = np.dot(kernel_mat[i], alpha[idx])
                if sign(pred_val) != y_for_now: 
                    if sign(pred_val) == 0:
                        alpha[idx][i] = y_for_now
                    alpha[idx][i] -= sign(pred_val)
        # In list of side target classifier, we transfer y_train[i] to -1
        if backward_idx != []:
            for idx in backward_idx:
                y_for_now = -1
                pred_val = np.dot(kernel_mat[i], alpha[idx])
                if sign(pred_val) != y_for_now:
                    if sign(pred_val) == 0:
                        alpha[idx][i] = y_for_now
                    alpha[idx][i] -= sign(pred_val)

    end = time.time()
    return mistakes, end-start, alpha



def test(X_train, X_test, y_test, alpha, num_class, kernel_mat, d):
    # This function aims to realize the testing process of perceptron (OvO)
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
    start = time.time()
    mistakes = 0
    calculate_mat = cal_mat(10)
    for i in range(len(X_test)):
        res = np.dot(alpha, kernel_mat[i]) #45*1
        res = np.array([1 if i > 0 else -1 for i in res]) #45*1
        res = np.dot(calculate_mat, res) #10*1 
        vote_res = np.argmax(res)
        if vote_res != y_test[i]:
            mistakes += 1
    end = time.time()
    return mistakes, end-start





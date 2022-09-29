import numpy as np
import time
from numpy.linalg import pinv as pinv
import matplotlib.pyplot as plt

def gen_data(m, n, winnow=False, knn=False):
    # This function aims to provide data for each algorithm
    '''
        Input:
            -m: int, number of data points
            -n: int, number of data dimension
            -winnow: bool, if winnow, we provide x in {0, 1}^n
            -knn: bool
        Output:
            X_train:  training data matrix with size m*n
            y_train:   training data label vector with size m*1
            X_test:    testing data matrix with size 10000*n
            y_test:    testing data label vector with size m*1
    '''
    if not winnow:
        X_train = np.random.randint(2, size=(m,n))*2-1 #size m*n
        X_test = np.random.randint(2, size=(10000,n))*2-1
    if winnow:
        X_train = np.random.randint(2, size=(m,n)) #size m*n
        X_test = np.random.randint(2, size=(10000,n))
    if knn:
        X_train = np.random.randint(2, size=(m,n))*2-1 #size m*n
        X_test = np.random.randint(2, size=(10000,n))*2-1
    y_train = X_train[:, 0]
    y_test = X_test[:, 0]
    
    return X_train, y_train, X_test, y_test



def sign(x):
    #sign function
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

def perceprtron(X_train, y_train, X_test, y_test):
    #This function aims to realize the function of perceptron
    '''
        Input:
            -X_train: training data matrix
            -y_train: training data label vector
            -X_test: testing data matrix
            -y_test: testing data label vector
        Output:
            -test_mistakes: int, mistakes on test data set
    '''

    data_mat_train = np.dot(X_train, np.transpose(X_train))#create data matrix
    data_mat_test = np.dot(X_test, np.transpose(X_train))
    #initialize alpha
    alpha = np.zeros(len(X_train))
    
    for i, x in enumerate(X_train):
        pred_val = np.dot(data_mat_train[i], alpha)
        if sign(pred_val) != y_train[i]:
            if sign(pred_val) == 0:
                alpha[i] = y_train[i]
            alpha[i] -= sign(pred_val)

    test_mistakes = 0
    for i, x in enumerate(X_test):
        pred_val = np.dot(data_mat_test[i], alpha)
        if sign(pred_val) != y_test[i]:
            test_mistakes += 1
    return test_mistakes/len(X_test)




def winnow(X_train, y_train, X_test, y_test):
    #This function aims to realize the function of winnow
    '''
        Input:
            -X_train: training data matrix
            -y_train: training data label vector
            -X_test: testing data matrix
            -y_test: testing data label vector
        Output:
            -test_mistakes: int, mistakes on test data set
    '''

    w_vector = np.ones(len(X_train[0]))
    # train_mistakes = 0
    for i, x in enumerate(X_train):
        pred_val = 0 if np.dot(w_vector, x) < len(x) else 1
        if y_train[i] != pred_val:
            # train_mistakes += 1
            for j in range(len(x)):
                w_vector[j] = w_vector[j] * 2**((y_train[i] - pred_val) * x[j] + 0.0)
        # print(train_mistakes)

    test_mistakes = 0
    for i, x in enumerate(X_test):
        pred_val = 0 if np.dot(w_vector, x) < len(x) else 1
        if pred_val != y_test[i]:
            test_mistakes += 1
    return test_mistakes/len(X_test)



def ls(X_train, y_train, X_test, y_test):
    #This function aims to realize the function of least square
    '''
        Input:
            -X_train: training data matrix
            -y_train: training data label vector
            -X_test: testing data matrix
            -y_test: testing data label vector
        Output:
            -test_mistakes: int, mistakes on test data set
    '''
    w = np.dot(np.matmul(pinv(np.matmul(X_train.T, X_train)), X_train.T), y_train)
    
    test_mistakes = 0
    for i, x in enumerate(X_test):
        pred_val = sign(np.dot(w, x))
        if pred_val != y_test[i]:
            test_mistakes += 1
    return test_mistakes/len(X_test)

    
def knn(X_train, y_train, X_test, y_test, k=1):
    #This function aims to realize the function of 1NN
    '''
        Input:
            -X_train: training data matrix
            -y_train: training data label vector
            -X_test: testing data matrix
            -y_test: testing data label vector
        Output:
            -test_mistakes: int, mistakes on test data set
    '''
    x_test_tile = np.reshape(np.tile(X_test,len(X_train)), (len(X_test),len(X_train),len(X_test[0])))#broadcast
    x_train_tile = np.reshape(np.tile(X_train,len(X_test)), (len(X_train),len(X_test),len(X_train[0])))#broadcast
    x_train_tile = x_train_tile.transpose(1,0,2)
    dist_mat = np.sum((x_test_tile - x_train_tile)**2, axis=2) 
    pred_idx = np.argmin(dist_mat, axis=1)
    pred_val = y_train[pred_idx[:]]
    return sum(abs(pred_val - y_test)/2)/len(y_test)



def test_alg(alg, dim_range, winnow=False):
    # This function aims to derive the sample complexity of each algorithm
    '''
        Input:
            -alg: algorithm function
            -dim_range: range, dimension range we want to test
            -winnow: optional bool, used for generating corresponding data
        Output:
            -m_list: list of sample complexity, each entry corresponds to dimension n.
    '''
    m_list = []
    for n in dim_range:
        m = 1
        while True:
            #generate data
            X_train, y_train, X_test, y_test = gen_data(m, n,  winnow=winnow)
            err = alg(X_train, y_train, X_test, y_test)
            if err <= 0.1:
                #if error < 10%, we repeat five times to ensure at 
                # least 4 times will converge to less than 10%
                times = 0
                for _ in range(5):
                    m_fix = m
                    X_train, y_train, X_test, y_test = gen_data(m_fix, n, winnow=winnow)
                    err_sub = alg(X_train, y_train, X_test, y_test)
                    if err_sub <= 0.1:
                        times += 1
                if times == 5:
                    m_list.append(m)
                    break
            m += 1
        print('finish run on n = ', n)
    return m_list


        
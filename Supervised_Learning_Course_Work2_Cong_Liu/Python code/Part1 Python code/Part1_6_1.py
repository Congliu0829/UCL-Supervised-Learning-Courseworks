
from Part1_utils_1_6 import *
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Reload data for task 1.6
    with open('X_zipcombo.npy', 'rb') as f:
        X = np.load(f)
    with open('y_zipcombo.npy', 'rb') as f:
        y = np.load(f)


    d_list = [i for i in range(1,7+1)]
    num_class = 10
    runs = 20



    train_error_rate_list = [[] for _ in range(len(d_list))]
    test_error_rate_list = [[] for _ in range(len(d_list))]

    for run in range(runs):
        print('Run: {0:}'.format(run))
        print('-------------------------------------------------------------------------------------------------\n')

        #split data set to 80% and 20%
        X_train, y_train, X_test, y_test = random_split(X, y)
    
        ##code test setting
        # X_train = X_train[:300]
        # y_train = y_train[:300]
        # X_test = X_test[:100]
        # y_test = y_test[:100]
        
        #initialize alpha
        alpha = np.zeros((len(d_list), int(num_class*(num_class-1)/2), len(X_train)))
        for d_idx, d in enumerate(d_list):
            print('Start training')
            print('---------------------------------------------------\n')
            train_total_error = 0
            test_total_error = 0
            store_box = [float('inf'), alpha[d_idx]]
            epochs = 0
            kernel_mat_train = kernel(X_train, np.transpose(X_train), d)
            kernel_mat_test = kernel(X_test, np.transpose(X_train), d)
            

            while epochs < 100:
                #train
                train_mistakes, times, alpha[d-1]= train(X_train, y_train, alpha[d-1], num_class, kernel_mat_train, d)
                print('Training - epoch{0:} required {1:.2f}s with {2:} mistakes out of {3:} items\n'.format(epochs, times, train_mistakes, len(X_train)))

                if train_mistakes < store_box[0]:
                    store_box[0], store_box[1] = train_mistakes, alpha[d-1]
                    epochs += 1
                else:
                    alpha[d-1] = store_box[1]
                    print('\nWe stop training the classifier with d = {0:} at epoch {1:}'.format(d, epochs))
                    break
            
            #getting training error
            train_mistakes, times, alpha[d-1]= train(X_train, y_train, alpha[d-1], num_class, kernel_mat_train, d)
            print('\n########################################')
            print('Start obtaining training error')
            print('Training on with d = {0:} -- It required {1:.2f}s with a train error of {2:.2f}%'.format(d, times, train_mistakes/len(X_train)*100))
            print('########################################\n')
            train_error_rate_list[d-1].append(train_mistakes/len(X_train))
            
            #testing
            test_mistakes, times = test(X_train, X_test, y_test, alpha[d-1], num_class, kernel_mat_test, d)
            print('\n########################################')
            print('Start testing')
            print('Testing on with d = {0:} -- It required {1:.2f}s with a test error of {2:.2f}%'.format(d, times, test_mistakes/len(X_test)*100))
            print('########################################\n')
            test_error_rate_list[d-1].append(test_mistakes / len(X_test))
            
            print('Finish training with d = {0:}'.format(d))
            print('---------------------------------------------------\n')
        

        print('Finish Run: {0:}'.format(run))
        print('-------------------------------------------------------------------------------------------------\n')


    print('Mean training error of 7 different d (%) (OvO)' + '\n')
    print(np.mean(train_error_rate_list, axis=1) * 100)

    print('Standard deviation of training error of 7 different d (%) (OvO)' + '\n')
    np.std(train_error_rate_list, axis=1) * 100

    print('Mean testing error of 7 different d (%) (OvO)' + '\n')
    print(np.mean(test_error_rate_list, axis=1) * 100)

    print('Standard deviation of testing error of 7 different d (%) (OvO)' + '\n')
    print(np.std(test_error_rate_list, axis=1) * 100)
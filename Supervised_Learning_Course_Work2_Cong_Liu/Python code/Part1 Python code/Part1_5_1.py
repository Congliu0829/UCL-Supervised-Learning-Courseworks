from Part1_utils_1_5 import *
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Reload data for task 1.5
    with open('X_zipcombo.npy', 'rb') as f:
        X = np.load(f)
    with open('y_zipcombo.npy', 'rb') as f:
        y = np.load(f)


    d_list = [10**i for i in range(-5, 5)]
    num_class = 10
    runs = 10


    train_error_rate_list = [[] for _ in range(len(d_list))]
    test_error_rate_list = [[] for _ in range(len(d_list))]

    for run in range(runs):
        print('Run: {0:}'.format(run))
        print('-------------------------------------------------------------------------------------------------\n')

        #split data set to 80% and 20%
        X_train, y_train, X_test, y_test = random_split(X, y)
        
        #initialize alpha
        alpha = np.array([[[0 for _ in range(len(X_train))] for _ in range(num_class)] for _ in range(len(d_list))])
        for d_idx, d in enumerate(d_list):
            print('Start training')
            print('---------------------------------------------------\n')
            train_total_error = 0
            test_total_error = 0
            store_box = [float('inf'), alpha[d_idx]]
            epochs = 0
            kernel_mat_train = K_matrix(X_train, X_train, d)
            kernel_mat_test = K_matrix(X_train, X_test, d)

            
            while epochs < 100:
                #train
                train_mistakes, times, alpha[d_idx]= train(X_train, y_train, alpha[d_idx], num_class, kernel_mat_train, d)
                print('Training - epoch{0:} required {1:.2f}s with {2:} mistakes out of {3:} items\n'.format(epochs, times, train_mistakes, len(X_train)))
                train_total_error += train_mistakes

                if train_mistakes < store_box[0]:
                    store_box[0], store_box[1] = train_mistakes, alpha[d_idx]
                    epochs += 1
                else:
                    alpha[d_idx] = store_box[1]
                    print('\nWe stop training the classifier with d = {0:} at epoch {1:}'.format(d, epochs))
                    train_error_rate_list[d_idx].append(train_total_error / epochs / len(X_train))
                    break
            #testing
            test_mistakes, times = test(X_train, X_test, y_test, alpha[d_idx], num_class, kernel_mat_test, d)
            print('\n########################################')
            print('Start testing')
            print('Testing on with d = {0:} -- It required {1:.2f}s with a test error of {2:.2f}%'.format(d, times, test_mistakes/len(X_test)*100))
            print('########################################\n')
            test_total_error += test_mistakes
            test_error_rate_list[d_idx].append(test_total_error / len(X_test))
            
            print('Finish training with d = {0:}'.format(d))
            print('---------------------------------------------------\n')
        

        print('Finish Run: {0:}'.format(run))
        print('-------------------------------------------------------------------------------------------------\n')


            


    print('Mean training error of 10 different c (%)' + '\n')
    print(np.mean(train_error_rate_list, axis=1) * 100)

    print('Standard Deviation of training error of 10 different d (%)' + '\n')
    print(np.std(train_error_rate_list, axis=1) * 100)

    print('Mean testing error of 10 different c (%)' + '\n')
    print(np.mean(test_error_rate_list, axis=1) * 100)

    print('Standard Deviation of testing error of 10 different d (%)' + '\n')
    print(np.std(test_error_rate_list, axis=1) * 100)
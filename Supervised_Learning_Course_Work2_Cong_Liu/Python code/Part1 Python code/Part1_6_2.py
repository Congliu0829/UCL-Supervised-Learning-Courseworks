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


    k_fold = 5
    d_list = [i for i in range(1,7+1)]
    num_class = 10
    runs = 20


    optimal_d_list = []
    train_total_list = []
    valid_total_list = []

    retrain_error_list = []


    for run in range(runs):
        print('Run: {0:}'.format(run))
        print('-------------------------------------------------------------------------------------------------\n')

        #split data set to 80% and 20%
        X_train_w, y_train_w, X_test, y_test = random_split(X, y)
        
        #split X_train, y_train to 5 subset and form a list for each of them
        X_train_list, y_train_list = cross_validation(X_train_w, y_train_w, k_fold)

        train_error_rate_list = [[0 for _ in range(k_fold)] for _ in range(len(d_list))]
        valid_error_rate_list = [[0 for _ in range(k_fold)] for _ in range(len(d_list))]

        print('Start K-fold cross validation')
        print('------------------------------------------------------------------------------\n')
        for k in range(k_fold):
            # obtain X_train, X_valid, y_train, y_valid in cross validation
            X_train_cv = []
            X_valid_cv = []
            y_train_cv = []
            y_valid_cv = []
            idx_list = [i for i in range(k_fold)]
            X_valid_cv.extend(X_train_list[k])
            y_valid_cv.extend(y_train_list[k])
            idx_list.remove(k)
            for idx in idx_list:
                X_train_cv.extend(X_train_list[idx])
                y_train_cv.extend(y_train_list[idx])
            
            X_train, X_valid, y_train, y_valid = X_train_cv, X_valid_cv, y_train_cv, y_valid_cv

            alpha = np.zeros((len(d_list), int(num_class*(num_class-1)/2), len(X_train)))
            for d in d_list:
                print('{0:}th fold: d = {1:}'.format(k, d))
                print('Start training')
                print('---------------------------------------------------\n')
                train_total_error = 0
                valid_total_error = 0
                store_box = [float('inf'), alpha[d-1]]
                epochs = 0
                #initialize alpha
                kernel_mat_train = kernel(X_train, np.transpose(X_train), d)
                kernel_mat_valid = kernel(X_valid, np.transpose(X_train), d)
                while epochs < 100:
                    #train
                    train_mistakes, times, alpha[d-1]= train(X_train, y_train, alpha[d-1], num_class, kernel_mat_train, d)
                    print('Training - epoch{0:} required {1:.2f}s with {2:} mistakes out of {3:} items\n'.format(epochs, times, train_mistakes, len(X_train)))
                    train_total_error += train_mistakes

                    if train_mistakes < store_box[0]:
                        store_box[0], store_box[1] = train_mistakes, alpha[d-1]
                        epochs += 1
                    else:
                        alpha[d-1] = store_box[1]
                        print('\nWe stop training the classifier with d = {0:} at epoch {1:}'.format(d, epochs))

                        train_error_rate_list[d-1][k] += train_total_error / epochs / len(X_train)
                        
                        break
                #validating
                valid_mistakes, times = test(X_train, X_valid, y_valid, alpha[d-1], num_class, kernel_mat_valid, d)
                print('\n########################################')
                print('Start validating')
                print('Validating on with d = {0:} on fold No.{1:} -- It required {2:.2f}s with a valid error of {3:.2f}%'.format(d, k, times, valid_mistakes/len(X_valid)*100))
                print('########################################\n')
                valid_total_error += valid_mistakes
                valid_error_rate_list[d-1][k] += valid_total_error / len(X_valid)
                
                print('Finish training with d = {0:}'.format(d))
                print('---------------------------------------------------\n')
        
        print('Finish K-fold cross validation')
        print('------------------------------------------------------------------------------\n')

        print('Finish Run: {0:}'.format(run))
        print('-------------------------------------------------------------------------------------------------\n')


        #select d that makes valid_error the smallest
        optimal_d = np.argmin(np.mean(valid_error_rate_list, axis = 1))+1
        optimal_d_list.append(optimal_d)

        train_total_list.append(train_error_rate_list)
        valid_total_list.append(valid_error_rate_list)



        #retrain on full 80% dataset using optimal_d in this run
        #initialization
        alpha = np.zeros((int(num_class*(num_class-1)/2), len(X_train_w)))
        epochs = 0
        store_box = [float('inf'), alpha]
        kernel_mat_train = kernel(X_train_w, np.transpose(X_train_w), optimal_d)
        while epochs < 100:
            train_mistakes, times, alpha = train(X_train_w, y_train_w, alpha, num_class, kernel_mat_train, optimal_d)
            print('Training on full data set - epoch{0:} required {1:.2f}s with {2:} mistakes out of {3:} items\n'.format(epochs, times, train_mistakes, len(X_train_w)))

            if train_mistakes < store_box[0]:
                store_box[0], store_box[1] = train_mistakes, alpha
                epochs += 1
            else:
                alpha = store_box[1]
                print('\nWe stop training the classifier with optimal d = {0:} at epoch {1:}'.format(optimal_d, epochs))
                break

        #testing
        kernel_mat_test = kernel(X_test, np.transpose(X_train_w), optimal_d)
        test_mistakes, times = test(X_train_w, X_test, y_test, alpha, num_class, kernel_mat_test, optimal_d)
        retrain_error_list.append(test_mistakes/len(X_test))
        print('\n########################################')
        print('Start testing')
        print('testing on with optimal d = {0:} -- It required {1:.2f}s with a test error of {2:.2f}%'.format(optimal_d, times, test_mistakes/len(X_test)*100))
        print('########################################\n')

            

    print('20 testing error with optimal d selected by k-fold cross validation (OvO)' + '\n')
    print(np.array(retrain_error_list))

    print('Mean of 20 testing error (%) with optimal d selected by k-fold cross validation (OvO)' + '\n')
    print(np.mean(retrain_error_list) * 100)

    print('Standard deviation of 20 testing error (%) with optimal d selected by k-fold cross validation (OvO)' + '\n')
    print(np.std(retrain_error_list) * 100)

    print('optimal d list selected by 20 runs (OvO)')
    print(optimal_d_list)

    print('Mean value of optimal d list selected by 20 runs (OvO)')
    print(np.mean(optimal_d_list))

    print('Standard deviation of optimal d list selected by 20 runs (OvO)')
    print(np.std(optimal_d_list))
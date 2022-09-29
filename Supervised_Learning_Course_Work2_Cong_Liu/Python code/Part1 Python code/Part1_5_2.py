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

    k_fold = 5
    d_list = [i*0.001 for i in range(10,31)]
    num_class = 10
    runs = 20


    optimal_d_list = []
    train_total_list = []
    valid_total_list = []

    retrain_error_list = []


    confusion_mat = np.zeros((runs, num_class, num_class)) 

    for run in range(runs):
        print('Run: {0:}'.format(run))
        print('-------------------------------------------------------------------------------------------------\n')
    
        #split data set to 80% and 20%
        X_train_w, y_train_w, X_test, y_test = random_split(X, y)

        # code test setting
        # X_train_w = X_train_w[:2000]
        # y_train_w = y_train_w[:2000]
        # X_test = X_test[:100]
        # y_test = y_test[:100]
        
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

            #calculate kernel_matrix in advance
            kernel_mat_train = [K_matrix(X_train, X_train, d) for d in d_list]
            kernel_mat_valid = [K_matrix(X_valid, X_train, d) for d in d_list]


            #initialize alpha for all
            alpha = np.array([[[0 for _ in range(len(X_train))] for _ in range(num_class)] for _ in range(len(d_list))])
            for d_idx, d in enumerate(d_list):
                print('{0:}th fold: d = {1:}'.format(k, d))
                print('Start training')
                print('---------------------------------------------------\n')
                train_total_error = 0
                valid_total_error = 0
                store_box = [float('inf'), alpha[d_idx]]
                epochs = 0

                #training
                while epochs < 100:
                    
                    train_mistakes, times, alpha[d_idx]= train(X_train, y_train, alpha[d_idx], num_class, kernel_mat_train[d_idx], d)
                    print('Training - epoch{0:} required {1:.2f}s with {2:} mistakes out of {3:} items\n'.format(epochs, times, train_mistakes, len(X_train)))
                    train_total_error += train_mistakes

                    if train_mistakes < store_box[0]:
                        store_box[0], store_box[1] = train_mistakes, alpha[d_idx]
                        epochs += 1
                    else:
                        alpha[d_idx] = store_box[1]
                        print('\nWe stop training the classifier with d = {0:} at epoch {1:}'.format(d, epochs))

                        train_error_rate_list[d_idx][k] += train_total_error / epochs / len(X_train)
                        
                        break
                #validating
                valid_mistakes, times = test(X_train, X_valid, y_valid, alpha[d_idx], num_class, kernel_mat_valid[d_idx], d)
                print('\n########################################')
                print('Start validating')
                print('Validating on with d = {0:} on fold No.{1:} -- It required {2:.2f}s with a valid error of {3:.2f}%'.format(d, k, times, valid_mistakes/len(X_valid)*100))
                print('########################################\n')
                valid_total_error += valid_mistakes
                valid_error_rate_list[d_idx][k] += valid_total_error / len(X_valid)
                
                print('Finish training with d = {0:}'.format(d))
                print('---------------------------------------------------\n')
        
        print('Finish K-fold cross validation')
        print('------------------------------------------------------------------------------\n')

        print('Finish Run: {0:}'.format(run))
        print('-------------------------------------------------------------------------------------------------\n')


        #select d that makes valid_error the smallest
        optimal_d = d_list[np.argmin(np.mean(valid_error_rate_list, axis = 1))]
        optimal_d_list.append(optimal_d)

        train_total_list.append(train_error_rate_list)
        valid_total_list.append(valid_error_rate_list)



        #retrain on full 80% dataset using optimal_d in this run
        #initialization setting
        alpha = np.array([[0 for _ in range(len(X_train_w))] for _ in range(num_class)])
        epochs = 0
        store_box = [float('inf'), alpha]
        kernel_mat_train = K_matrix(X_train_w, X_train_w, optimal_d)
        kernel_mat_test = K_matrix(X_train_w, X_test, optimal_d)
        
        
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
        test_mistakes, times, confusion_mat[run] = test_confusion(X_train_w, X_test, y_test, alpha, num_class, kernel_mat_test, optimal_d, confusion_mat[run])
        retrain_error_list.append(test_mistakes/len(X_test))
        print('\n########################################')
        print('Start testing')
        print('testing on with optimal d = {0:} -- It required {1:.2f}s with a test error of {2:.2f}%'.format(optimal_d, times, test_mistakes/len(X_test)*100))
        print('########################################\n')

            
    print('Mean of 20 testing error (%) with optimal c selected by k-fold cross validation' + '\n')
    print(np.mean(retrain_error_list))

    print('20 testing error (%) with optimal c selected by k-fold cross validation' + '\n')
    print(np.array(retrain_error_list)*100)

    print('Standard deviation of 20 testing error (%) with optimal c selected by k-fold cross validation' + '\n')
    np.std(retrain_error_list)

    print('optimal c list selected by k fold cross validation')
    print(optimal_d_list)

    print('mean optimal c over 20 runs')
    print(np.mean(optimal_d_list))

    print('Standard deviation of optimal c over 20 runs')
    print(np.std(optimal_d_list))


    print('Start comparing between Polynomial kernel and Gaussian kernel')
    d_list = [0.01645]
    num_class = 10
    runs = 10
    total_epochs = 0
    total_time = 0

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
        alpha = np.array([[[0 for _ in range(len(X_train))] for _ in range(num_class)] for _ in range(len(d_list))])
        for d_idx, d in enumerate(d_list):
            print('Start training')
            print('---------------------------------------------------\n')
            store_box = [float('inf'), alpha[d_idx]]
            epochs = 0
            time_k = 0
            kernel_mat_train = K_matrix(X_train, X_train, d)
            kernel_mat_test = K_matrix(X_train, X_test, d)
            while epochs < 100:
                #train
                train_mistakes, times, alpha[d_idx]= train(X_train, y_train, alpha[d_idx], num_class, kernel_mat_train, d)
                print('Training - epoch{0:} required {1:.2f}s with {2:} mistakes out of {3:} items\n'.format(epochs, times, train_mistakes, len(X_train)))
                time_k += times

                if train_mistakes < store_box[0]:
                    store_box[0], store_box[1] = train_mistakes, alpha[d_idx]
                    epochs += 1
                else:
                    alpha[d_idx] = store_box[1]
                    print('\nWe stop training the classifier with d = {0:} at epoch {1:}'.format(d, epochs))
                    total_epochs += epochs
                    total_time += time_k / epochs
                    break
        

        print('Finish Run: {0:}'.format(run))
        print('-------------------------------------------------------------------------------------------------\n')


            
    print('Averaged epoches trained to be converged with Gaussian Kernel using optimal c' + '\n')
    print(total_epochs/10)

    print('Averaged time spent for one epoch with Gaussian Kernel using optimal c:' + '\n')
    print(total_time/10)

    def kernel(x1, x2, d):
        return (np.dot(x1, x2))**d
        
    d_list = [5]
    num_class = 10
    runs = 10
    total_epochs = 0
    total_time = 0

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
            store_box = [float('inf'), alpha[d_idx]]
            epochs = 0
            time_k = 0
            # kernel_mat_train = K_matrix(X_train, X_train, d)
            # kernel_mat_test = K_matrix(X_train, X_test, d)
            kernel_mat_train = kernel(X_train, np.transpose(X_train), d)
            kernel_mat_test = kernel(X_test, np.transpose(X_train), d)
            while epochs < 100:
                #train
                train_mistakes, times, alpha[d_idx]= train(X_train, y_train, alpha[d_idx], num_class, kernel_mat_train, d)
                print('Training - epoch{0:} required {1:.2f}s with {2:} mistakes out of {3:} items\n'.format(epochs, times, train_mistakes, len(X_train)))
                time_k += times

                if train_mistakes < store_box[0]:
                    store_box[0], store_box[1] = train_mistakes, alpha[d_idx]
                    epochs += 1
                else:
                    alpha[d_idx] = store_box[1]
                    print('\nWe stop training the classifier with d = {0:} at epoch {1:}'.format(d, epochs))
                    total_epochs += epochs
                    total_time += time_k / epochs
                    break
        

        print('Finish Run: {0:}'.format(run))
        print('-------------------------------------------------------------------------------------------------\n')


    print('Averaged epoches trained to be converged with Polynomial Kernel using optimal d:' + '\n')
    print(total_epochs/10)

    print('Averaged time spent for one epoch with Polynomial Kernel using optimal d:' + '\n')
    print(total_time/10)
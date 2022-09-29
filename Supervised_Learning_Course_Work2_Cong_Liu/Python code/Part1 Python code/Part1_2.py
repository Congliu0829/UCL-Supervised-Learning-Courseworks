from Part1_utils_from_1_1_to_1_4 import random_split, cross_validation, train, test, test_confusion
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Reload data for task 1.2
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

    #initialize the record box
    #record_box[i][0] is the # of wrong prediction
    #record_box[i][1] is the output of wrong prediction
    global record_box
    record_box = np.zeros((len(X), 2))


    confusion_mat = np.zeros((runs, num_class, num_class)) #20*10*10的矩阵 20个10*10的confusion matrix
    for run in range(runs):
        print('Run: {0:}'.format(run))
        print('-'*100 + '\n')

        #split data set to 80% and 20%
        X_train_w, y_train_w, X_test, y_test = random_split(X, y)
        
        #code test setting
        X_train_w = X_train_w[:300]
        y_train_w = y_train_w[:300]
        X_test = X_test[:100]
        y_test = y_test[:100]
        
        #split X_train, y_train to 5 subset and form a list for each of them
        X_train_list, y_train_list = cross_validation(X_train_w, y_train_w, k_fold)

        train_error_rate_list = [[0 for _ in range(k_fold)] for _ in range(len(d_list))]
        valid_error_rate_list = [[0 for _ in range(k_fold)] for _ in range(len(d_list))]

        print('Start K-fold cross validation')
        print('-'*50 + '\n')
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



            alpha = np.array([[[0 for _ in range(len(X_train))] for _ in range(num_class)] for _ in range(len(d_list))])
            for d in d_list:
                print('{0:}th fold: d = {1:}'.format(k, d))
                print('Start training')
                print('-'*20 + '\n')
                train_total_error = 0
                valid_total_error = 0
                store_box = [float('inf'), alpha[d-1]]
                epochs = 0
                #initialize alpha
                
                while epochs < 100:
                    #train
                    train_mistakes, times, alpha[d-1] = train(X_train, y_train, alpha[d-1], num_class, d)
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
                valid_mistakes, times = test(X_train, X_valid, y_valid, alpha[d-1], num_class, d)
                print('\n' + '#'*20)
                print('Start validating')
                print('Validating on with d = {0:} on fold No.{1:} -- It required {2:.2f}s with a valid error of {3:.2f}%'.format(d, k, times, valid_mistakes/len(X_valid)*100))
                print('#'*20 + '\n')
                valid_total_error += valid_mistakes
                valid_error_rate_list[d-1][k] += valid_total_error / len(X_valid)
                
                print('Finish training with d = {0:}'.format(d))
                print('-'*20 + '\n')
        
        print('Finish K-fold cross validation')
        print('-'*50 + '\n')

        print('Finish Run: {0:}'.format(run))
        print('-'*100 + '\n')

        #select d that makes valid_error the smallest
        optimal_d = np.argmin(np.mean(valid_error_rate_list, axis = 1))+1
        optimal_d_list.append(optimal_d)

        train_total_list.append(train_error_rate_list)
        valid_total_list.append(valid_error_rate_list)



        #retrain on full 80% dataset using optimal_d in this run
        #initialization
        alpha = np.array([[0 for _ in range(len(X_train_w))] for _ in range(num_class)])
        epochs = 0
        store_box = [float('inf'), alpha]
        
        while epochs < 100:
            train_mistakes, times, alpha = train(X_train_w, y_train_w, alpha, num_class, optimal_d)
            print('Training on full data set - epoch{0:} required {1:.2f}s with {2:} mistakes out of {3:} items\n'.format(epochs, times, train_mistakes, len(X_train_w)))

            if train_mistakes < store_box[0]:
                store_box[0], store_box[1] = train_mistakes, alpha
                epochs += 1
            else:
                alpha = store_box[1]
                print('\nWe stop training the classifier with optimal d = {0:} at epoch {1:}'.format(optimal_d, epochs))
                break
        
        #1.2 & 1.3
        #testing
        test_mistakes, times, confusion_mat[run] = test_confusion(X_train_w, X_test, y_test, alpha, num_class, optimal_d, confusion_mat[run])
        retrain_error_list.append(test_mistakes/len(X_test))
        print('\n' + '#'*20)
        print('Start testing')
        print('testing on with optimal d = {0:} -- It required {1:.2f}s with a test error of {2:.2f}%'.format(optimal_d, times, test_mistakes/len(X_test)*100))
        print('#'*20 + '\n')

        #1.4
        #retrain using optimal_d on full dataset and 
        #record the output as well as the times it gets wrong
        alpha = np.array([[0 for _ in range(len(X))] for _ in range(num_class)])
        epochs = 0
        store_box = [float('inf'), alpha]

        while epochs < 100:
            train_mistakes, times, alpha = train(X, y, alpha, num_class, optimal_d)
            print('Training on full data set to find 5 hardest label- epoch{0:} required {1:.2f}s with {2:} mistakes out of {3:} items\n'.format(epochs, times, train_mistakes, len(X)))

            if train_mistakes < store_box[0]:
                store_box[0], store_box[1] = train_mistakes, alpha
                epochs += 1
            else:
                alpha = store_box[1]
                print('\nWe stop training the classifier with optimal d = {0:} at epoch {1:}'.format(optimal_d, epochs))
                break
        
        #testing
        test_mistakes, times, record_box = test(X, X, y, alpha, num_class, optimal_d, record_box, record=True)

    # 1.2 print information
    print('20 testing error with optimal d selected by k-fold cross validation' + '\n')
    print(np.array(retrain_error_list))
    print('Mean value of 20 testing error with optimal d selected by k-fold cross validation' + '\n')
    print(np.mean(retrain_error_list))
    print('Standard deviation of 20 testing error with optimal d selected by k-fold cross validation' + '\n')
    print(np.std(retrain_error_list))
    print('optimal d list' + '\n')
    print(optimal_d_list)
    print('Mean value of optmial d list')
    print(np.mean(optimal_d_list))
    print('Standard deviation of optimal d list')
    print(np.std(optimal_d_list))

    # Save files for latter experiments
    np.save("confusion_matrix.npy", confusion_mat)
    np.save("record_box.npy", record_box)

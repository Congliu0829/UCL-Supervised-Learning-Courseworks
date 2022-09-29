from Part1_utils_from_1_1_to_1_4 import random_split, train, test
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    # Load data
    print('Start loading data ...')
    df = pd.read_csv('zipcombo.dat.txt',sep='\t',header=None)
    data = df.values
    y = np.array([int(float(data[i][0].split(" ")[0])) for i in range(len(data))])
    X = np.array([[float(data[i][0].split(" ")[j]) for j in range(1, len(data[i][0].split(" ")) - 1)] for i in range(len(data))])

    print('Data Summary:\nlength of data: {0:}\nsize of X: {1:}'.format(len(X), len(X[0])))
    print('Example of single data pair X,y\nX:\n', X[0],"\ny:", y[0])
    
    # Save X and y for the latter question
    np.save("X_zipcombo.npy", X)
    np.save("y_zipcombo.npy", y)

    # Experiment 1.1
    ##### one classifier run on one dataset for 20 times
    d_list = [i for i in range(1,7+1)]
    num_class = 10
    runs = 20


    train_error_rate_list = [[] for _ in range(len(d_list))]
    test_error_rate_list = [[] for _ in range(len(d_list))]

    for run in range(runs):
        print('Run: {0:}'.format(run))
        print('-'*100 + '\n')

        #split data set to 80% and 20%
        X_train, y_train, X_test, y_test = random_split(X, y)
        
        
        #initialize alpha
        alpha = np.array([[[0 for _ in range(len(X_train))] for _ in range(num_class)] for _ in range(len(d_list))])
        for d in d_list:
            print('Start training')
            print('-'*50 + '\n')
            store_box = [float('inf'), alpha[d-1]]
            epochs = 0
            
            while epochs < 100:
                #train
                train_mistakes, times, alpha[d-1]= train(X_train, y_train, alpha[d-1], num_class, d)
                print('Training - epoch{0:} required {1:.2f}s with {2:} mistakes out of {3:} items\n'.format(epochs, times, train_mistakes, len(X_train)))

                if train_mistakes < store_box[0]:
                    store_box[0], store_box[1] = train_mistakes, alpha[d-1]
                    epochs += 1
                else:
                    alpha[d-1] = store_box[1]
                    print('\nWe stop training the classifier with d = {0:} at epoch {1:}'.format(d, epochs))
                    break
            
            #getting training error
            train_mistakes, times, alpha[d-1]= train(X_train, y_train, alpha[d-1], num_class, d)
            print('\n' + '#'*20)
            print('Start obtaining training error')
            print('Training on with d = {0:} -- It required {1:.2f}s with a train error of {2:.2f}%'.format(d, times, train_mistakes/len(X_train)*100))
            print('#'*30 + '\n')
            train_error_rate_list[d-1].append(train_mistakes/len(X_train))
            
            #testing
            test_mistakes, times = test(X_train, X_test, y_test, alpha[d-1], num_class, d)
            print('\n' + '#'*20)
            print('Start testing')
            print('Testing on with d = {0:} -- It required {1:.2f}s with a test error of {2:.2f}%'.format(d, times, test_mistakes/len(X_test)*100))
            print('#'*30 + '\n')
            test_error_rate_list[d-1].append(test_mistakes / len(X_test))
            
            print('Finish training with d = {0:}'.format(d))
            print('-'*40 + '\n')
        

        print('Finish Run: {0:}'.format(run))
        print('-'*100 + '\n')


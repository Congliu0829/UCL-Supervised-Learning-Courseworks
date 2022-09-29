import numpy as np
import matplotlib.pyplot as plt
from Question6 import train_test_data
from Question6 import label_mat
from Question6 import gen_error_list
from Question6 import gen_data

# Define specific gen_error_list for Ex8, with an extra output optimal_k
def gen_error_list(X_h, y_h, train_size, k_list):
    '''
    Input: -X_h: array, 100 data points generated uniform randomly.

           -y_h: array, corresponded labels of X_h

           -train_size: array, list containing sizes of training set that we want to test

           -k_list: array, list containing number of k that we want to test

    Output: -error_list: array, list containing generalization error list, 
                         each entry represents the generalization error 
                         averaged on 100 runs using specified k

            -optimal_k: array, list containing optimal k value for each of the training size,
                        averaged on 100 runs.
    '''
    error_list = np.zeros(len(k_list))
    optimal_k = 0
    for epoch in range(100):
        # 1. sample h  h = knn(X_h, y_h, x_test, 3) 100, 3
        # 2. generate 4000 training data and 1000 testing data
        X_train, y_train = train_test_data(train_size, X_h, y_h, noise=0.2)
        X_test, y_test = train_test_data(1000, X_h, y_h, noise=0.2)
        # 3. error evaluation
        label_matrix = label_mat(X_train, y_train, X_test)
        k_error_list = []

        for k in k_list:
            k_error = 0
            for i in range(len(y_test)):
                y_pred = np.argmax(np.bincount(label_matrix[i][:k]))
                if y_pred != y_test[i]:
                    k_error += 1
            k_error_list.append(k_error / len(y_test))
        optimal_k += np.argmin(k_error_list) + 1
        error_list += np.array(k_error_list)
    optimal_k /= 100
    error_list /= 100

    return error_list, optimal_k


# paramter initialization for Ex8
X_h, y_h = gen_data(100)
train_size_list = [100]
sub_list = [500 * i for i in range(1, 9)]
train_size_list.extend(sub_list)
optimal_list = []
k_list = [i for i in range(1, 50)]

# run process of Ex8
for train_size in train_size_list:
    _, optimal_k = gen_error_list(X_h, y_h, train_size, k_list)
    optimal_list.append(optimal_k)

# plot and save the figure for Ex8
plt.plot(train_size_list, optimal_list)
plt.xlabel("number of training samples m")
plt.ylabel("optimal K")
plt.grid()
plt.savefig("2-8", dpi=500)
plt.show()

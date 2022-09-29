import numpy as np
import time
from numpy.linalg import pinv as pinv
import matplotlib.pyplot as plt
from Part3_utils import *

if __name__ == '__main__':
    print("start obtaining sample complexity")
    runs = 20
    dim_range = np.arange(1,101)


    all_m_list_precep = []
    all_m_list_winnow = []
    all_m_list_ls = []

    for run in range(runs):
        m_list_precep = test_alg(perceprtron, dim_range)
        m_list_winnow = test_alg(winnow, dim_range, winnow=True)
        m_list_ls = test_alg(ls, dim_range)

        all_m_list_precep.append(m_list_precep)
        all_m_list_winnow.append(m_list_winnow)
        all_m_list_ls.append(m_list_ls)


    runs = 20
    dim_range = np.arange(1,15+1)

    all_m_list_knn = []
    for run in range(runs):
        print('Run ', run)
        m_list_knn = []
        m = 1
        for n in dim_range:
            flag_knn = 0
            while flag_knn== 0:
                #generate data
                X_train, y_train, X_test, y_test = gen_data(m, n, knn=True)
                #1nn
                if flag_knn < 1:
                    err_knn = knn(X_train, y_train, X_test, y_test)
                    if err_knn <= 0.1:
                        m_list_knn.append(m)
                        flag_knn += 1 
                m += 1
            print('finish run on n = ', n)
        
        all_m_list_knn.append(m_list_knn)

    print('sample complexity of perceptron' + "\n")
    plt.errorbar(np.arange(1,101),np.mean(all_m_list_precep, axis=0),yerr = np.std(all_m_list_precep, axis=0), label='Perceptron')
    plt.xlabel("n, dimension of data")
    plt.ylabel("m, number of data")
    plt.grid()
    plt.legend()
    plt.savefig('3-1-perceptron.png', dpi=500)
    plt.show()

    print('sample complexity of winnow' + "\n")
    plt.errorbar(np.arange(1,101),np.mean(all_m_list_winnow, axis=0),yerr = np.std(all_m_list_winnow, axis=0), label='Winnow')
    plt.xlabel("n, dimension of data")
    plt.ylabel("m, number of data")
    plt.grid()
    plt.legend()
    plt.savefig('3-1-winnow.png', dpi=500)
    plt.show()

    print('sample complexity of ls' + "\n")
    plt.errorbar(np.arange(1,101),np.mean(all_m_list_ls, axis=0),yerr = np.std(all_m_list_ls, axis=0), label='Least Square')
    plt.xlabel("n, dimension of data")
    plt.ylabel("m, number of data")
    plt.grid()
    plt.legend()
    plt.savefig('3-1-ls.png', dpi=500)
    plt.show()

    print('sample complexity of 1NN' + "\n")
    plt.errorbar(np.arange(1,16),np.mean(all_m_list_knn, axis=0),yerr = np.std(all_m_list_knn, axis=0), label='1NN')
    plt.xlabel("n, dimension of data")
    plt.ylabel("m, number of data")
    plt.grid()
    plt.legend()
    plt.savefig('3-1-1nn.png', dpi=500)
    plt.show()

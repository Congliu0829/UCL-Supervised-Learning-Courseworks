import numpy as np
from numpy.linalg import norm, eig
import matplotlib.pyplot as plt
import pandas as pd


# Functions for the task 3
def W_matrix(xdata, c):
    '''
    With W_ij = exp(-c||x_i-x_j||^2), we get the W matrix.

    input:
        -xdata: array (m*n), the input data
        -c: float, parameter c
    output:
        -W: array (m*m), the weight matrix
    '''
    num_samples, dim_samples = xdata.shape[0], xdata.shape[1]
    W = np.zeros((num_samples, num_samples))
    for row in range(W.shape[0]):
        for col in range(W.shape[1]):
            W[row][col] = np.exp(-c*norm(xdata[row]-xdata[col])**2)
    return W

def L_matrix(W):
    '''
    Define D as the diagonal matrix which D_ii=sum of i-th row of W.
    Then we get graph laplacian L = D - W

    input:
        -W: array(m*m), the weight matrix
    output:
        -L: array(m*m), the graph laplacian
    '''
    diagonal_sum = np.sum(W, axis=1)
    L = -W
    for i in range(len(W)):
        L[i][i] = diagonal_sum[i] - W[i][i]
    return L

def cluster(L):
    '''
    Compute eigenvalues and eigenvectors of L, then sort eigenvalues. Pick 
    the eigenvector with the second smallest eigenvalue. Predict y_i by 
    determining the sign of i-th element of the eigenvector.

    input:
        -L: array(m*m), the graph laplacian
    output:
        -y_hat: array(1*m), our predicted y
    '''
    eigenvalue, eigenvector = eig(L)
    eig_dict = {eigenvalue[i]:eigenvector[:,i] for i in range(len(eigenvalue))}
    v2 = eig_dict[sorted(eig_dict)[1]]
    y_hat = np.zeros(len(L))
    for i in range(len(L)):
        if v2[i] >= 0:
            y_hat[i] = 1
        else:
            y_hat[i] = -1
    return y_hat

if __name__ == '__main__':
    # Initialize dataframe and some variables
    df = pd.read_csv('dtrain123.dat', header=None)
    num_samples = df.shape[0]
    x_dim = len(df.iloc[0][0].split()) - 1
    x, y = [], []

    # Get data with labels '1' and '3'
    for i in range(num_samples):
        data = df.iloc[i][0].split()
        if float(data[0]) == 1 or float(data[0]) == 3:
            y.append(int(float(data[0])))
            x.append([float(data[j+1]) for j in range(x_dim)])
    x, y = np.array(x), np.array(y)
    new_num_samples = len(y)

    # Change labels from {1,3} to {-1,1} for the convenience of comparison
    classy = np.zeros(new_num_samples)
    for i in range(new_num_samples):
        classy[i] = 2 * (y[i]==1) - 1

    # With different c, get CP score
    cp_list = []
    c_list = np.arange(0,0.1001,0.001)
    for c in c_list:
        W = W_matrix(x, c)
        L = L_matrix(W)
        y_hat = cluster(L)
        l_pos, l_neg = 0, 0
        for i in range(new_num_samples):
            if y_hat[i] == classy[i]:
                l_pos += 1
            else:
                l_neg += 1
        cp_list.append(max(l_pos, l_neg) / new_num_samples)
    cp_list = np.array(cp_list)
    opt_c = c_list[np.argmax(cp_list)]

    # Plot the figure
    plt.plot(c_list, cp_list)
    plt.xlabel("c value")
    plt.ylabel("CP score")
    plt.title('CP vs c with optimal c={}'.format(opt_c))
    plt.grid()
    plt.savefig("2-3.png",dpi=800)


    #####--------- Attention -----------#####
    # Below part is code for the analysis   #
    # of question 4. we plot a 3d scatter   #
    # for a better understanding.           #

    from mpl_toolkits.mplot3d import Axes3D

    # Get sorted eigenvalue list with different c values
    eigenvalue_list = []
    c_list = np.arange(0,0.1001,0.001)
    for c in c_list:
        W = W_matrix(x, c)
        L = L_matrix(W)
        eigenvalue, eigenvector = eig(L)
        norm_eigenvalue = np.zeros(len(eigenvalue))
        for i in range(len(eigenvalue)):
            norm_eigenvalue[i] = norm(eigenvalue[i])
        norm_eigenvalue = sorted(norm_eigenvalue)
        eigenvalue_list.append(norm_eigenvalue)
    eigenvalue_list = np.array(eigenvalue_list)

    # Flatten x and y aixs
    c_tile = np.tile(c_list, len(eigenvalue))
    x_axis = np.reshape(c_tile, (len(eigenvalue), len(c_list))).transpose(1,0).flatten()
    index_tile = np.tile(np.arange(1,len(eigenvalue)+1), len(c_list))
    y_axis = np.reshape(index_tile, (len(c_list), len(eigenvalue))).flatten()

    # Plot the figure
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax = Axes3D(fig)
    ax.scatter(x_axis, y_axis, eigenvalue_list, s=1)
    ax.set_xlabel('c value', fontsize=9, rotation=150)
    ax.set_ylabel('ordered i-th eigenvalue', fontsize=9, rotation=150)
    ax.set_zlabel('eigenvalue', fontsize=9)
    plt.savefig('2-3-b.png', dpi=500)
    plt.show()
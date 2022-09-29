import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, eig

# Functions for the task 1
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

def plot_best_figure(x, y, c_list, task=1):
    '''
    With given inputs, labels, and parameter c list, choose c which could (almost)
    correctly cluster data. Then plot the clustering figure. Finally return the 
    optimal c and related predicted y.

    input:
        -x: array (m*n), the input data
        -y: array (1*m), the label data
        -c_list: list, a list of c values
        -task: int, number of task in case to save the figure
    output:
        -opt_c: float, optimal c which could (almost) correctly cluster data
        -opt_y_hat: the predicted-y list with optimal c.
    '''
    num_samples = len(x)
    min_wrong_c = num_samples

    for c in c_list:
        W = W_matrix(x, c)
        L = L_matrix(W)
        y_hat = cluster(L)
        error = 0
        # If c could result a smaller error number, then update opt_c
        for i in range(num_samples):
            if y_hat[i] != y[i]:
                error += 1
        if error < min_wrong_c:
            min_wrong_c = error
            opt_c = c
            opt_y_hat = y_hat

    # Plot the correctly clustered data
    group1, group2 = [], []
    for i in range(len(opt_y_hat)):
        if opt_y_hat[i] == -1:
            group1.append(x[i])
        else:
            group2.append(x[i])
    group1, group2 = np.array(group1), np.array(group2)

    fig, ax = plt.subplots()
    ax.scatter(group1[:,0], group1[:,1], label='class -1')
    ax.scatter(group2[:,0], group2[:,1], label='class +1')
    plt.title('Spectral clustering algorithm with c={:.3f}'.format(opt_c))
    plt.grid()
    plt.legend()
    plt.savefig("2-"+str(task)+"-2.png",dpi=500)
    plt.show()

    return opt_c, opt_y_hat

if __name__ == '__main__':
    ### Task 1
    # Get initial data points
    df = pd.read_csv('twomoons.dat', header=None)
    num_samples = df.shape[0]
    x_dim = len(df.iloc[0][0].split()) - 1

    y = np.array([int(float(df.iloc[i][0].split()[0])) for i in range(num_samples)])
    x = np.array([[float(df.iloc[j][0].split()[i]) for i in range(1,x_dim+1)] for j in\
                range(num_samples)])

    plt.scatter(x[:,0],x[:,1])
    plt.title('Original data')
    plt.grid()
    plt.savefig('2-1-1.png', dpi=500)
    plt.show()

    # Get optimal c and plot the clustered data
    c_list = [2**i for i in np.arange(-10,10.01,0.1)]
    opt_c1, opt_y_hat1 = plot_best_figure(x, y, c_list, task=1)
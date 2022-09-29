#package initialization for Question1
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Define functions for all the rest of exercise 1
def phi_x(x, k):
    '''
    For a given input and degree k, return its feature map with 
        k-degree polynomial basis

    Input: -x: array, input training data

           -k: int, degree of the basis

    Output: array but change each element in the original x to an
                array with k elements 
    '''
    feature_mapped_x = [[] for i in range(len(x))]
    for i in range(len(x)):
        feature_mapped_x[i] = [(x[i] ** m) for m in range(0, k)]
    return np.array(feature_mapped_x)


def weight(x, g):
    '''
    Compute w = (X_T·X)^(-1)X_T·y

    Input: -x: array, feature map of x

           -g: array, y data

    Output: array with dimention of features
    '''
    return np.matmul(np.matmul(inv(np.matmul(x.T, x)), x.T), g)


def cal_mse(g, g_predict, k):
    '''
    Compute mse with given y data and predicted y data

    Input: -g: array, y data

           -g_predict: array, predicted y data

           -k: int, degree of basis

    Output: int, the value of mse
    '''
    mse = 0
    for i in range(len(g)):
        mse += (g[i] - g_predict[i]) ** 2
    return mse / len(g)


def predict(x, g, k):
    '''
    Get predicted y data and weight. First get feature map of x, then calculate
        the weight, and finally get y_hat = phi·weight
    
    Input: -x: array, training input

           -g: array, training label

    Output: -y_hat: array, predicted y with training data

            -w: array, dimention equal to features
    '''
    phix = phi_x(x, k)
    w = weight(phix, g)
    return np.matmul(phix, w), w


def test(x, k, w):
    '''
    Get predicted y with testing inputs and trained weight

    Input: -x: array, testing input

           -k: int, degree of basis

           -w: array, trained weight

    Output: array, predicted test y
    '''
    phix_test = phi_x(x, k)
    return np.matmul(phix_test, w)

#### 1-(a) #####################################################################################
# Initialization of data
k_list = [1,2,3,4]
x = np.array([1,2,3,4])
y = np.array([3,2,0,5])
plot_x = np.arange(0,4.3,1e-3) # points for ploting curve

w_list, y_hat_plot_list = [], []
for k in k_list:
    y_hat, w = predict(x, y, k)
    w_list.append(w)
    y_hat_plot_list.append(test(plot_x, k, w))


fig, ax = plt.subplots()
ax.scatter(x, y, label='data')
for i in range(len(k_list)):
    ax.plot(plot_x, y_hat_plot_list[i], label='predicted curve with k={}'.format(k_list[i]))
plt.legend()
plt.xlim([0,4.5])
plt.grid()
plt.savefig("1-1-a",dpi=500)
plt.show()

##### 1-(b) #####################################################################################
print(w_list)
# # the list of w with different degree k. Each element is the coordinates from degree 0 to k

# print("k=1, y=2.5")
# print("k=2, y=1.5+0.4x")
# print("k=3, y=9-7.1x+1.5x^2")
# print("k=4, y=-5+15.17x-8.5x^2+1.33x^3")

##### 1-(c) #####################################################################################
mse_list = []
for i in range(4):
    y_hat = test(x, k_list[i], w_list[i])
    mse_list.append(cal_mse(y, y_hat, k_list[i]))
print(mse_list) # The list of mse with different k



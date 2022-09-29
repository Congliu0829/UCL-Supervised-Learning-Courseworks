# Initialization for librarys and k lists
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
k_list1 = [2, 5, 10, 14, 18]
k_list2 = [i for i in range(1, 19)]

# Define functions for all the rest of exercise 2
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


def plotcurve(x, g, x_plot, g_predict, k):
    '''
    Plot the trained curve with weight

    Input: -x: array, training input

           -g: array, training label

           -x_plot: array, generated x points waiting to be predicted

           -g_predict: array, predicted training y

           -k: int, degree of basis
    '''
    # Pair x and y_predict, then sort the points
    g_dict, g_predict_dict = {}, {}
    for i in range(len(x)):
        g_dict[x[i]] = g[i]
        g_predict_dict[x[i]] = g_predict[i]
    for i in range(len(x)):
        g[i] = g_dict[x[i]]
        g_predict[i] = g_predict_dict[x[i]] 

    fig, ax = plt.subplots()
    ax.scatter(x, g, label='data points', c  = 'r')
    ax.plot(x_plot, g_predict, label='predicted function', c = "b")
    plt.title('k = {0}'.format(k))
    plt.legend()
    plt.grid()
    plt.savefig("1-2-a-ii_" + str(k))
    plt.show()


##### 2-(a)-i #########################################################################################################################################################################
# Data-point initialization
from matplotlib.pyplot import figure
x = np.random.uniform(0, 1, 30)
epsilon = np.random.normal(0, 0.07, 30)
g = (np.sin(2 * np.pi * x))**2 + epsilon

# Print data points and functions
fig, ax = plt.subplots()
ax.scatter(x, g, label='randomly generated data points')
function_x = np.arange(0, 1, 0.0001)
ax.plot(function_x, (np.sin(2 * np.pi * function_x))**2, label='g(x)')

plt.grid()
plt.xlim([0,1])
plt.xlabel('x')
plt.ylabel('g(x)')
plt.legend()
plt.savefig("1-2-a-i",dpi=500)
plt.show()


##### 2-(a)-ii #########################################################################################################################################################################
# Plot different curves with k's
for k in k_list1:
	g_predict, w = predict(x, g, k)
	x_plot = np.arange(0,1,0.0001)
	g_plot = test(x_plot, k, w)
	plotcurve(x, g, x_plot, g_plot, k)


##### 2-(b) #########################################################################################################################################################################
# Plot log(mse) for different k
mse_list = []
for k in k_list2:
    g_predict, w = predict(x, g, k)
    mse = cal_mse(g, g_predict, k)
    mse_list.append(mse)
plt.plot(k_list2, np.log(mse_list), label='log(mse)')
plt.legend()
plt.grid()
plt.show()


##### 2-(c) #########################################################################################################################################################################
# Initialize test-pool datapoints
x_test = np.random.uniform(0, 1, 1000)
epsilon_test = np.random.normal(0, 0.07, 1000)
g_test = (np.sin(2 * np.pi * x_test))**2 + epsilon_test

mse_test_list = []
for k in k_list2:
    g_predict, w = predict(x, g, k)
    g_predict_test = test(x_test, k, w)
    test_mse = cal_mse(g_test, g_predict_test, k)
    mse_test_list.append(test_mse)
plt.plot(k_list2, np.log(mse_test_list), label='log(mse)')
plt.legend()
plt.grid()
plt.show()


##### 2-(d) #########################################################################################################################################################################
# repeat 2-(b) and 2-(c) with 100 epochs
mse_train_list = []
mse_test_list = []

for k in k_list2:
    mse_train = 0
    mse_test = 0
    epoch = 0
    while epoch < 100:
        # Generate data for each epoch
        x = np.random.uniform(0, 1, 30)
        epsilon = np.random.normal(0, 0.07, 30)
        g = (np.sin(2 * np.pi * x))**2 + epsilon
        x_test = np.random.uniform(0, 1, 1000)
        epsilon_test = np.random.normal(0, 0.07, 1000)
        g_test = (np.sin(2 * np.pi * x_test))**2 + epsilon_test
        ### train and get trained mse
        g_predict, w = predict(x, g, k)
        g_predict_test = test(x_test, k, w)
        mse_train += cal_mse(g_predict, g, k)
        mse_test += cal_mse(g_test, g_predict_test, k)
        epoch += 1
    # Get the average mse values
    mse_train_list.append(mse_train/100)
    mse_test_list.append(mse_test/100)

# Plot curves with ln(avg)
plt.plot(k_list2, np.log(mse_train_list), label='train')
plt.plot(k_list2, np.log(mse_test_list), label='test')
plt.legend()
plt.xlabel("Polynomial dimensions")
plt.ylabel("ln(Mean Squared Error)")
plt.grid()
plt.savefig("1-2-d", dpi=500)
plt.show()
	
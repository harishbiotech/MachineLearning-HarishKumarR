#read csv file
import pandas as pd
import numpy as np
from scipy.special.cython_special import huber
from sklearn.metrics import r2_score
df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')

# X and y - disease_score_fluct
X_train = df.iloc[:48,:5]
y_train = df.iloc[:48,6:7]
X_test = df.iloc[49:,:5]
y_test = df.iloc[49:,6:7]
#function to compute the hypothesis
X_np_train = X_train.values
m=X_np_train.shape[0]
X_with_bias_train = np.concatenate((np.ones((m,1)), X_np_train), axis=1)
theta=np.zeros((X_with_bias_train.shape[1]))
theta_matrix=theta.reshape(6,1)
print("Theta",theta_matrix)
print("x test =",X_with_bias_train)
y_np_train = y_train.values.reshape(-1, 1)
print("y test =",y_np_train)
#------------------------step1---------------------------------------------#
#--------set theta value as 0 and compute hypothesis-----------------------#
#------------------------hypothesis----------------------------------------#
def hypothesis(X_with_bias_train, theta_matrix):
    return np.dot(X_with_bias_train, theta_matrix)
#----------hypothesis is calculated at 1st all hypothesis is 0 ------------#
def cost(h, y_np):
    for i in range(len(y_np_train)):
        s = 0
        s += ((hypothes[i][0] - y_np[i][0]) ** 2)
    return s*1/2
h=hypothesis(X_with_bias_train,theta_matrix)
print("h =",h)
print("len h =",len(h[0]))
alpha=0.000001127
print("alpha",alpha)
#---------Now find theta with alpha and theta and hypothesis-----------------#

def find_theta(X_with_bias_train,y_np_train,h,alpha,theta_matrix):
    thet=[]
    for i in range(len(X_with_bias_train[0])):
        s=0
        for j in range(len(y_np_train)):
            s+=(h[j][0]-y_np_train[j][0])*X_with_bias_train[j][i]    # summation

        thet.append(theta_matrix[i][0]-(alpha*s))        # multiply by alpha and subtract with theta

    theta=np.array(thet)
    return theta.reshape(-1,1)


theta=find_theta(X_with_bias_train,y_np_train,h,alpha,theta_matrix)
print(theta)
print("-"*30)
#
#
#
#
for i in range(1000):
    Theta=theta
    hypothes = hypothesis(X_with_bias_train,Theta)
    prev_cost = cost(hypothes, y_np_train)
    print(i,"cost",prev_cost)
    theta_update = find_theta(X_with_bias_train,y_np_train,hypothes,alpha,Theta)
    print(i,"theta_updated",theta_update)
    print(i,"hypothesis",hypothes)
    Theta=theta_update
print("-"*30)
theta_to_test = theta_update
print("theta_to_test",theta_to_test)
print("-"*30,"Testing the model","-"*90)
X_np_test = X_test.values
m=X_np_test.shape[0]
X_with_bias_test = np.concatenate((np.ones((m,1)), X_np_test), axis=1)
# theta=np.zeros((X_with_bias_test.shape[1]))
# theta_matrix=theta.reshape(6,1)
# print("Theta",theta_matrix)
print("x test =",X_with_bias_test)
y_np_test = y_test.values.reshape(-1, 1)
print("y test =",y_np_test)
Prediction_testing=hypothesis(X_with_bias_test,theta_to_test)
print("Predicted y values = ",Prediction_testing)
r_square_value=r2_score(y_np_test,Prediction_testing)
print("r_square_value = ",r_square_value)
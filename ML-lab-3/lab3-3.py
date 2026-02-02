#read csv file
import pandas as pd
import numpy as np
from scipy.special.cython_special import huber
import math
df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')

# X and y - disease_score_fluct
X = df.iloc[:,:5]
y = df['disease_score_fluct']

# function to compute the hypothesis
X_np = X.values
m=X_np.shape[0]
X_with_bias = np.concatenate((np.ones((m,1)), X_np), axis=1)
theta=np.zeros((X_with_bias.shape[1]))
theta_matrix=theta.reshape(6,1)
print("Theta",theta_matrix)
print("x =",X_with_bias)
y_np = y.values.reshape(-1, 1)

print("y =",y_np)
# def hypothesis(X_with_bias, theta_matrix):
#     hypo = []
#     for i in range(X_with_bias.shape[0]):  # loop over samples
#         s = 0
#         for j in range(X_with_bias.shape[1]):  # loop over features
#             s += theta_matrix[j][0] * X_with_bias[i][j]
#         hypo.append(s)
#     return np.array(hypo)
#------------------------step1---------------------------------------------#
#--------set theta value as 0 and compute hypothesis-----------------------#
#------------------------hypothesis----------------------------------------#
def hypothesis(X_with_bias, theta_matrix):
    return np.dot(X_with_bias, theta_matrix)
#----------hypothesis is calculated at 1st all hypothesis is 0 ------------#

def cost(h, y_np):
    for i in range(len(y_np)):
        s = 0
        s += ((hypothes[i][0] - y_np[i][0]) ** 2)
    return s*1/2
h=hypothesis(X_with_bias,theta_matrix)
print("h =",h)
print("len h =",len(h[0]))
#print("x",X_with_bias[0][1])
alpha=0.000000924
print("alpha",alpha)
#---------Now find theta with alpha and theta and hypothesis-----------------#
#---------failed theta-------------------------------------------------------# don't use 3 for loop this is not matric multiplication
# def find_theta(X_with_bias,y_np,h,alpha,theta_matrix):
#     thet=[]
#     for i in range(len(X_with_bias[0])):
#         a=[]
#         for j in range(len(y)):
#             s=0
#             for k in range(len(h[0])):
#                 s+=(h[j][k]-y_np[j][k])*X_with_bias[j][i]
#             a.append(theta_matrix[i][0]-(alpha*s))
#         thet.append(a)
    #theta=thet.value.reshape(6,1)
    # return thet
#----------------second attempt----------------------------------------------#
def find_theta(X_with_bias,y_np,h,alpha,theta_matrix):
    thet=[]
    for i in range(len(X_with_bias[0])):
        s=0
        for j in range(len(y_np)):
            s+=(h[j][0]-y_np[j][0])*X_with_bias[j][i]    # summation

        thet.append(theta_matrix[i][0]-(alpha*s))        # multiply by alpha and subtract with theta

    theta=np.array(thet)
    return theta.reshape(-1,1)


theta=find_theta(X_with_bias,y_np,h,alpha,theta_matrix)
print(theta)
print("-"*30)




for i in range(5):
    Theta=theta
    hypothes = hypothesis(X_with_bias,Theta)
    prev_cost = cost(hypothes, y_np)
    print(i,"cost",prev_cost)
    theta_update = find_theta(X_with_bias,y_np,hypothes,alpha,Theta)
    print(i,"theta_updated",theta_update)
    print(i,"hypothesis",hypothes)
    Theta=theta_update




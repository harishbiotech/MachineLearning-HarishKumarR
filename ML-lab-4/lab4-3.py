#-----------------Normal method implementation from scratch-----------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
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

X_np_train = X_train.values
theta=np.zeros((X_np_train.shape[1]))
theta_matrix=theta.reshape(5,1)
# print("Theta",theta_matrix)
# print("x test =",X_np_train)
y_np_train = y_train.values.reshape(-1, 1)
# print("y test =",y_np_train)
#------------------------step1---------------------------------------------#
#--------set theta value as 0 and compute hypothesis-----------------------#
#------------------------hypothesis----------------------------------------#
def hypothesis(X_np_train, theta_matrix):
    return np.dot(X_np_train, theta_matrix)
# h=hypothesis(X_train, theta_matrix)
# print("hypothesis",h)
def normal_equation(X_np_train, y_np_train):
    XT = np.transpose(X_np_train)
    XTX = np.matmul(XT, X_np_train)
    XTX_inv = np.linalg.inv(XTX)
    XTX_invXT = np.matmul(XTX_inv, XT)
    XTX_invXTy = np.matmul(XTX_invXT, y_np_train)
    theta = XTX_invXTy
    return theta
ne=normal_equation(X_np_train, y_np_train)
print("normal equation",ne)
second_hypothesis = hypothesis(X_np_train, ne)
print("second hypothesis",second_hypothesis)
r2_value = r2_score(y_np_train, second_hypothesis)
print("r2 own code disease score fluct",r2_value)
#-------------------------------------------------------------
print("-----------------------Admission predict------------------------------------------------")
df = pd.read_csv('Admission_Predict.csv')

# X and y - disease_score_fluct
X_train = df.iloc[:320,:8]
y_train = df.iloc[:320,8:9]
X_test = df.iloc[321:,:8]
y_test = df.iloc[321:,8:9]
X_np_train = X_train.values
y_np_train = y_train.values.reshape(-1, 1)
X_np_test = X_test.values
y_np_test = y_test.values
# print("X_train =",X_np_train)
# print("y_train =",y_np_train)
# print("X_test =",X_np_test)
# print("y_test =",y_np_test)
theta=np.zeros((X_np_train.shape[1]))
theta_matrix=theta.reshape(8,1)
# print("Theta = ",theta_matrix)
def hypothesis(X_np_train, theta_matrix):
    return np.dot(X_np_train, theta_matrix)
# h=hypothesis(X_train, theta_matrix)
# print("hypothesis",h)
def normal_equation(X_np_train, y_np_train):
    XT = np.transpose(X_np_train)
    XTX = np.matmul(XT, X_np_train)
    XTX_inv = np.linalg.inv(XTX)
    XTX_invXT = np.matmul(XTX_inv, XT)
    XTX_invXTy = np.matmul(XTX_invXT, y_np_train)
    theta = XTX_invXTy
    return theta
ne=normal_equation(X_np_train, y_np_train)
print("normal equation",ne)
second_hypothesis = hypothesis(X_np_train, ne)
print("second hypothesis",second_hypothesis)
r2_value = r2_score(y_np_train, second_hypothesis)
print("r2 own code admission predict",r2_value)


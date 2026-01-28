#read csv file
import pandas as pd
import numpy as np
from scipy.special.cython_special import huber

df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')

# X and y - disease_score_fluct
X = df.iloc[:,:5]
y = df['disease_score_fluct']

# function to compute the hypothesis
X_np = X.values
m=X_np.shape[0]
X_with_bias = np.concatenate((np.ones((m,1)), X_np), axis=1)

def hypothesis(X_with_bias,theta):
   return np.dot(X_with_bias, theta)
def cost():

#Data normalization - scale the values between 0 and 1. Implement code from scratch.
import numpy as np
import pandas as pd

df=pd.read_csv('data_kfold.csv')
X = df.drop(columns=[df.columns[0], df.columns[1], df.columns[32]])
y = df['diagnosis']
y = y.map({'M': 1, 'B': 0})

#====Final try===============
def normalize(X):
    maximum = X.max(axis=0)
    minimum = X.min(axis=0)
    X_scaled = (X - minimum) / (maximum - minimum)
    return X_scaled
X = normalize(X)
print(X)

#try one failed because of accessing elements
# def normalize(X):
#     for i in range(X.shape[0]):
#         maximum=X.iloc[:,0].max()
#         minimum=X.iloc[:,0].min()
#         X_scaled=(X.iloc[0:i,0]-minimum)/(maximum-minimum)
#         return X_scaled
# print(normalize(X))

#try two failed beacause this is for only one column
# def normalize(X):
#     maximum = X.iloc[:,0].max()
#     minimum = X.iloc[:,0].min()
#     X_scaled = (X.iloc[:,0] - minimum) / (maximum - minimum)
#     return X_scaled

#try three failed beacuse cannot applied to all the columns in the X
# for i in range(X.shape[1]):
#     X_test_scaled=normalize(X.iloc[:,i])
#     print(X_test_scaled)
#Try four directly apply the function into X

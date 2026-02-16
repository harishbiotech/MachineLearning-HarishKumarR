#Data standardization - scale the values such that mean of new dist = 0 and sd = 1. Implement code from scratch.
import pandas as pd

df=pd.read_csv('data_kfold.csv')
X = df.drop(columns=[df.columns[0], df.columns[1], df.columns[32]])
y = df['diagnosis']
y = y.map({'M': 1, 'B': 0})

#This i did with the same idea which i use in lab 6 2nd question
def standardize(X):
    mean = X.mean()
    std = X.std()
    X_std = (X - mean) / std
    return X_std
X_standartized = standardize(X)
print(X_standartized)

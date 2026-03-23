#Implement information gain measures. The function should accept data points for parents, data points for
# both children and return an information gain value.
import numpy as np
import pandas as pd
import math

df = pd.read_csv("ENTROPY.csv")

X = df.drop(columns=["play"])
y = df["play"]


def Entropy(y):

    probs = y.value_counts(normalize=True)

    s = 0

    for p in probs:
        s += -p * math.log(p, 2)

    return s


def information_gain(X, y):

    H_parent = Entropy(y)

    print("Parent Entropy:", H_parent)

    for col in X.columns:

        print(f"\nColumn: {col}")

        weighted_entropy = 0

        for val in X[col].unique():

            indices = X[X[col] == val].index
            #print('indices',indices)
            y_val = y.loc[indices]
            #print('y_val',y_val)
            H_child = Entropy(y_val)
            #print('H_child',H_child)
            weight = len(y_val) / len(y)
            #print('weight',weight)
            weighted_entropy += weight * H_child
           # print('weighted_entropy',weighted_entropy)
           # print(f"{val} -> Entropy: {H_child}, Weight: {weight}")

        IG = H_parent - weighted_entropy

        print(f"Information Gain ({col}): {IG}")

print(information_gain(X, y))
# def information_gain(X, y):
#
#     for col in X.columns:
#
#         print(f"\nColumn: {col}")
#
#         for val in X[col].unique():
#
#             indices = X[X[col] == val].index
#
#             y_val = y.loc[indices]
#
#             print(f"Value: {val} -> {y_val}")
#
#             y_val_uniq = y_val.unique().tolist()
#
#             for val in y_val_uniq:
#
#                 H = entropy(val)
#
#             return H
#
#
# information_gain(X, y)

# def information_gain(X,y):
#
#     def unique_values(X):
#
#         c=[]
#         for col in X.columns:
#             u=(X[col].unique())
#
#             for val in u:
#                 c.append(val)
#         return c
#
#     u = unique_values(X)
#
#     for col in X.columns:
#         for val in u:
#             i = X[X[col] == val].index
#             y_val = y[:i,0].tolist()
#             print(y_val)
#
# print(information_gain(X,y))














#=================================================================================================================
# c=[]
# for col in X.columns:
#     u=(X[col].unique())
#
#     for val in u:
#         c.append(val)
# print(c)
#
#
# col = X["weather"].values
#
# for val in np.unique(col):
#     indices = np.where(col == val)[0]
#     print(type(indices))
#     print(indices)
# for col in X.columns:
#     c=[]
#     a=(X[col].unique())
#     c.append(col,':',a)
#     uniq.append(c)
# print(uniq)
# def information_gain(X,y):
#     for col in X.columns:
#         unique_values = []
#         unique_values.append(X[col].unique())
#     return unique_values
#
# print(information_gain(X,y))
#Implement entropy measure using Python.
#The function should accept a set of data points and their class labels and return the entropy value.
import math
import pandas as pd
import numpy as np

df = pd.read_csv("ENTROPY.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
def Entropy(y):
    y_unique=y.value_counts(normalize=True)
    print(y_unique)
    proportions = y_unique.tolist()
    entropy=[]
    s = 0
    for i in range(len(proportions)):
        s=s+(-proportions[i])*(math.log(proportions[i],2))
    entropy.append(s)
    return entropy
print(Entropy(y))

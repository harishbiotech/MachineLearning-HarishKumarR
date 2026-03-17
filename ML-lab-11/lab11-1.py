import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math

def load_data():
    df = pd.read_csv("Iris.csv")
    X=df.drop(columns=['Id','Species'])
    y=df['Species'].map({ "Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2})
    return X,y

def root_entropy(X,y):
    y_unique=y.value_counts(normalize=True)
    print(y_unique)
    proportions=y_unique.to_list()
    print(proportions)
    entropy=[]
    s=0
    for i in range(len(proportions)):
        s = s + (-proportions[i]) * (math.log(proportions[i], 3))
    entropy.append(s)
    return entropy

def Information_gain(X,y):
    X=X.tolist()
    print(X)

X,y=load_data()
print(root_entropy(X,y))


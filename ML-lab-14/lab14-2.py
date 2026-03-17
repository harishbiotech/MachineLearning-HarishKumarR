import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():

    df = pd.read_csv("Iris.csv")

    X = df.drop(columns=["Id","Species"])
    y = df["Species"]

    y = y.map({
        "Iris-setosa":0,
        "Iris-versicolor":1,
        "Iris-virginica":2
    })

    return X.values,y.values


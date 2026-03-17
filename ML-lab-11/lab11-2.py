import pandas as pd
import numpy as np

df = pd.read_csv("Iris.csv")
X = df.drop(columns=['Id', 'Species'])
y = df['Species'].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
X = X.sort_values(by=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
for i in range(len(X.columns)):
    for j in range(len(X.iloc[:,:1])):
        a=(X.iloc[:j,:i]+X.iloc[:j+1,:i])/2




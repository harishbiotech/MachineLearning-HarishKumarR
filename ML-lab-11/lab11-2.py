import pandas as pd
import numpy as np

df = pd.read_csv("Iris.csv")
X = df.drop(columns=['Id', 'Species'])
y = df['Species'].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
X = X.sort_values(by=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
print(X)

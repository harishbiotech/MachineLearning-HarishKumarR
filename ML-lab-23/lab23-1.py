import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.miscmodels import count

df = pd.read_csv('Iris.csv')

X=df[['SepalLengthCm','SepalWidthCm']]
y=df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

#calculate the prior probability

# total = y_train.value_counts()
# print(total)
# P_Iris_setosa = total['Iris-setosa']/(total['Iris-setosa']+total['Iris-versicolor']+total['Iris-virginica'])
# print(P_Iris_setosa) this is manual way of computing each probaility
#there is a function in pandas for probaility .value_count(Normalize=True)

probabilities_y = y_train.value_counts(normalize=True)

#calculating likelihood

L_sestosa = X_train[y_train=='Iris-setosa'].value_counts()
# print(X_train['SepalLengthCm'].value_counts())
print(L_sestosa)
print(min(X_train['SepalLengthCm']))
print(max(X_train['SepalLengthCm']))
print(min(X_train['SepalWidthCm']))
print(max(X_train['SepalWidthCm']))
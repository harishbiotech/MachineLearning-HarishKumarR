import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


df = pd.read_csv("sonar.csv")
X=df.iloc[:,:-1]
y=df.iloc[:,-1].map({'R':0,'M':1})
X_np=X.values
y_np=y.values
k=int(input("Enter the k value:"))
s=X_np.shape


for i in range(k):

    start = i * (s[0] // k)
    end = i * (s[0] // k) + (s[0] // k)

    if k-i==1:
        X_test=X_np[start:]
        y_test=y_np[start:]
        X_train = X_np[:start,]
        y_train = y_np[:start,]

    else:
        X_test = X_np[start:end]
        y_test = y_np[start:end]
        X_train = np.append(X_np[:start,], X_np[end:,], axis=0)
        y_train = np.append(y_np[:start,], y_np[end:,], axis=0)

        print(f'{start} : {end}')


        model=LogisticRegression()
        model.fit(X_train,y_train)
        prediction=model.predict(X_test)
        accuracy=accuracy_score(y_test,prediction)
        print(f'{i} split Accuracy : {accuracy*100}%')
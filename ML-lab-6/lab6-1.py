#K-fold cross validation. Implement for K = 10. Implement from scratch, then, use scikit-learn methods.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv('data_kfold.csv')
X = df.drop(columns=[df.columns[0], df.columns[1], df.columns[32]])
y = df['diagnosis']
y = y.map({'M': 1, 'B': 0})
X_np = X.values
y_np = y.values
s=X_np.shape
k=int(input("Enter the k value:"))
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    prediction = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, prediction)
    print(f"{i} split  Accuracy score is : {accuracy}"
            "  "f"Accuracy percentage is : {accuracy*100}%")


#==============Implement Stochastic Gradient Descent algorithm from scratch==================
#==============                                                            ==================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
X = df.drop(columns=[df.columns[0], df.columns[1], df.columns[32]])
y = df['diagnosis']
y = y.map({'M': 1, 'B': 0})
X_np = X.values
y_np = y.values


def split_data(X_np, y_np):
    X_train,X_test,y_train,y_test=train_test_split(X_np, y_np, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = split_data(X_np, y_np)

# ---------------- STANDARDIZATION ----------------
def standardize(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
X_train_scaled, X_test_scaled = standardize(X_train, X_test)
y_train_np= np.asarray(y_train).reshape(-1,1)
y_test_np= np.asarray(y_test).reshape(-1,1)
theta=np.zeros(X_train_scaled[0].shape).reshape(-1,1)
alpha=0.001
def hypothesis(theta, X_train_scaled):
    z = np.dot(X_train_scaled, theta)
    hypothesis = 1/(1+np.exp(-z))
    return hypothesis
h=hypothesis(theta, X_train_scaled)
def Stochastic_Gradient_Descent(theta, y_train_np, hypothesis, X_test_scaled,alpha):
    thet=[]
    for i in range(len(X_test_scaled[0])):
        s=0
        for j in range(len(y_train_np)):
            s+=((y_train_np[j][0]-hypothesis[j][0])*X_train_scaled[j][i])
        thet.append(theta[i][0]+(alpha*s))
    return np.array(thet).reshape(-1, 1)
theta_updated=Stochastic_Gradient_Descent(theta, y_train_np, h, X_train_scaled,alpha)
theta_Transpose=theta_updated
X_multiply=np.matmul(X_train_scaled, theta_Transpose)
X_multiply=np.sort(X_multiply, axis=0)
print(X_multiply)
Y=1/(1+np.exp(-X_multiply))
print(Y)
plt.title('sigmoid')
plt.plot(X_multiply,Y)
plt.show()
w=(1/1+np.exp(-X_multiply))*(1-(1/1+np.exp(-X_multiply)))
plt.title('differentiated sigmoid')
plt.plot(X_multiply,w)
plt.show()





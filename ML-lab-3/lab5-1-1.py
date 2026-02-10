import numpy as np

X_train_scaled=np.array([[1,0],[2,1],[3,1],[4,2],[5,3]])
y=np.array([[0],[0],[0],[1],[1]])
theta=np.zeros(X_train_scaled[0].shape).reshape(-1,1)
alpha=0.1
def hypothesis(theta, X_train_scaled):
    z = np.dot(X_train_scaled, theta)
    hypothesis = 1/(1+np.exp(-z))
    return hypothesis
h=hypothesis(theta, X_train_scaled)
print(h)
def Stochastic_Gradient_Descent(theta, y, hypothesis, X_train_scaled,alpha):
    thet=[]
    for i in range(len(X_train_scaled[0])):
        s=0
        for j in range(len(y)):
            s+=((y[j][0]-hypothesis[j][0])*X_train_scaled[j][i])
        thet.append(theta[i][0]+(alpha*s))
    return np.array(thet).reshape(-1,1)
theta_updated=Stochastic_Gradient_Descent(theta, y, h, X_train_scaled,alpha)
print(theta_updated)
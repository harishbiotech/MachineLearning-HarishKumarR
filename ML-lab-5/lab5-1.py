#==============Implement Stochastic Gradient Descent algorithm from scratch==================
#==============                                                            ==================
import numpy as np


def Stochastic_Gradient_Descent(theta, y, hypothesis, X,alpha):
    thet=[]
    for i in range(len(X[0])):
        s=0
        for j in range(len(y)):
            s+=((y[j][0]-hypothesis[j][0])*X[i][j])
        thet.append(theta[0][i]+(alpha*s))
    return theta.reshape(-1,1)



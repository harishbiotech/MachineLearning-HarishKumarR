import pandas as pd
import numpy as np
import math

X1=np.array([3,6])
X2=np.array([10,10])
dot=np.dot(X1,X2)
print(dot)

def transform(X1,X2):
    return(X1**2,math.sqrt(2)*X1*X2,X2**2)

print(transform(X1,X2))
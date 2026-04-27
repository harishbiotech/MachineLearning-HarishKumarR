import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from transform.py import Transform

# data = [
#     [1,13,"Blue"],[1,18,"Blue"],[2,9,"Blue"],[3,6,"Blue"],
#     [6,3,"Blue"],[9,2,"Blue"],[13,1,"Blue"],[18,1,"Blue"],
#     [3,15,"Red"],[6,6,"Red"],[6,11,"Red"],[9,5,"Red"],
#     [10,10,"Red"],[11,5,"Red"],[12,6,"Red"],[16,3,"Red"]
# ]

# X = np.array([[d[0], d[1]] for d in data])
# y = np.array([d[2] for d in data])
X = np.array([3,6])
y = np.array([10,10])

def Transform(X,y):
    # x1 = X[:, 0]
    # x2 = X[:, 1]
    return (X**2)*(y**2)+2*

x_trans = Transform(X,y)
X_dot = np.dot(X,y)**2
print(X_dot)
print(x_trans)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from transform.py import Transform

data = [
    [1,13,"Blue"],[1,18,"Blue"],[2,9,"Blue"],[3,6,"Blue"],
    [6,3,"Blue"],[9,2,"Blue"],[13,1,"Blue"],[18,1,"Blue"],
    [3,15,"Red"],[6,6,"Red"],[6,11,"Red"],[9,5,"Red"],
    [10,10,"Red"],[11,5,"Red"],[12,6,"Red"],[16,3,"Red"]
]

X = np.array([[d[0], d[1]] for d in data])
y = np.array([d[2] for d in data])

def Transform(X):
    x1 = X[:, 0]
    x2 = X[:, 1]

    return np.column_stack((
        x1 ** 2,
        np.sqrt(2) * x1 * x2,
        x2 ** 2
    ))
X_trans = Transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for label in ["Blue", "Red"]:
    pts = X_trans[y == label]
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], label=label)

ax.set_xlabel("x1^2")
ax.set_ylabel("√2 x1x2")
ax.set_zlabel("x2^2")

plt.legend()
plt.title("Transformed 3D Data")
plt.show()

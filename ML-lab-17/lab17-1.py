
import numpy as np
import matplotlib.pyplot as plt

# Data
data = [
    [1,13,"Blue"],[1,18,"Blue"],[2,9,"Blue"],[3,6,"Blue"],
    [6,3,"Blue"],[9,2,"Blue"],[13,1,"Blue"],[18,1,"Blue"],
    [3,15,"Red"],[6,6,"Red"],[6,11,"Red"],[9,5,"Red"],
    [10,10,"Red"],[11,5,"Red"],[12,6,"Red"],[16,3,"Red"]
]

X = np.array([[d[0], d[1]] for d in data])
y = np.array([d[2] for d in data])

# Plot
for label in ["Blue", "Red"]:
    points = X[y == label]
    plt.scatter(points[:,0], points[:,1], label=label)

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Original 2D Data")
plt.show()
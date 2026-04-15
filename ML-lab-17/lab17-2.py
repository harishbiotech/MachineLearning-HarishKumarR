import pandas as pd
import numpy as np
import math

X1=np.array([3,6])
X2=np.array([10,10])
k=[]
for i in range(0,len(X1)):
    k.append(X1[i]**2+X2[i]**2+2)























# Convert labels to colors
# color_map = {"Blue": "blue", "Red": "red"}
# colors = y.map(color_map)
#
# # Plotting
# fig, ax = plt.subplots(figsize=(5, 4))
#
# # Correct min/max (column-wise)
# x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
# y_min, y_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()
#
# ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
#
# # Correct indexing using .iloc
# scatter = ax.scatter(
#     X.iloc[:, 0],
#     X.iloc[:, 1],
#     s=150,
#     c=colors,
#     edgecolors="k"
# )
#
# # Legend (clean way)
# for label in y.unique():
#     ax.scatter([], [], c=color_map[label], label=label)
#
# ax.legend(title="Classes")
#
# ax.set_title("Samples in two-dimensional feature space")
# ax.set_xlabel("x1")
# ax.set_ylabel("x2")
#
# plt.show()
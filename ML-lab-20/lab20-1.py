import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
print(X)
# Standardize
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Apply PCA (all components)
pca = PCA()
X_pca = pca.fit_transform(X)

# Explained variance
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Scree Plot
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("Scree Plot (Cumulative Variance)")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.grid()
plt.show()

# Reduce to 2 components
pca_2 = PCA(n_components=2)
X_reduced = pca_2.fit_transform(X)

# Plot PCA result
plt.figure()
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title("PCA (2 Components)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
#
# data = load_iris()
# X = pd.DataFrame(data.data, columns=data.feature_names)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# pca = PCA()
# X_pca = pca.fit_transform(X_scaled)
#
# print("Explained Variance Ratio:")
# print(pca.explained_variance_ratio_)
#
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
# plt.title("Scree Plot (Cumulative Variance)")
# plt.xlabel("Number of Components")
# plt.ylabel("Cumulative Variance")
# plt.grid()
# plt.show()
#
# pca_2 = PCA(n_components=2)
# X_reduced = pca_2.fit_transform(X_scaled)
#
# plt.figure()
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
# plt.title("PCA (2 Components)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid()
# plt.show()
#
# kmeans = KMeans(n_clusters=3, random_state=1)
# labels = kmeans.fit_predict(X_scaled)
#
#
# plt.figure()
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
# plt.title("K-Means Clustering (K=3)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid()
# plt.show()
#
# inertia = []
# K_range = range(1, 10)
# for k in K_range:
#     km = KMeans(n_clusters=k, random_state=1)
#     km.fit(X_scaled)
#     inertia.append(km.inertia_)
#
# plt.figure()
# plt.plot(K_range, inertia, marker='o')
# plt.title("Elbow Method")
# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("Inertia")
# plt.grid()
# plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy.state import scale
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, cut_tree
from ISLP.cluster import compute_linkage


def q1():
    us = get_rdataset("USArrests").data
    scale = StandardScaler(with_std=True,with_mean=True)
    us_scaled = scale.fit_transform(us)
    pca = PCA()
    pca_us = pca.fit_transform(us_scaled)
    score = pca.transform(us_scaled)
    lables=['PC1','PC2','PC3','PC4']
    plt.bar(lables,pca.explained_variance_ratio_,label='Explained Variance')
    plt.plot(lables,pca.explained_variance_ratio_,color='red',label='Explained Variance',marker='D')
    plt.plot(lables,pca.explained_variance_ratio_.cumsum(),color='orange',label='Cumulative Explained Variance',marker='X')
    plt.legend()
    plt.show()

def q2():
    np.random.seed(0);
    X = np.random.standard_normal((50, 2));
    X[:25, 0] += 3;
    X[:25, 1] -= 4;
    kmeans = KMeans(n_clusters=2,random_state=2,n_init=20).fit(X)
    kmeans.labels_
    plt.scatter(X[:,0],X[:,1],c=kmeans.labels_)
    plt.show()

def main():
    q1()
    q2()

if __name__ == '__main__':
    main()

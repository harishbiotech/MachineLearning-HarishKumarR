from sklearn.metrics import accuracy_score
from ISLP import load_data
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils._repr_html import estimator


def Load_data():
    data = load_data('NCI60')
    df = data['data']
    y = data['labels']
    X = pd.DataFrame(df)
    return X,y

def pca(X,y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=44)
    pca_x = pca.fit_transform(X_scaled)

    X_label = []
    for i in range(1,45):
        X_label.append(f"PC{i}")

    plt.bar(X_label, pca.explained_variance_ratio_, label='Explained Variance')
    plt.plot(X_label, pca.explained_variance_ratio_.cumsum(), color='orange', label='Cumulative Explained Variance',
             marker='X')
    plt.legend()
    plt.show()
    return pca_x

def pca_logisticregression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)
    pca_model = PCA(n_components=44)
    pca_x_train = pca_model.fit_transform(X_train_scale)#pca_model 2 models are using
    pca_x_test = pca_model.transform(X_test_scale)
    model = LogisticRegression(max_iter=1000)
    model.fit(pca_x_train, y_train)
    y_pred = model.predict(pca_x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def hierarchical_clustering(X, y):
    scaler = StandardScaler()

X,y = Load_data()
pca_pca = pca(X,y)
pca_logisticregression(X,y)




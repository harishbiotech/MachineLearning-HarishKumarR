from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

def load_data(path):
    df = pd.read_csv('Iris.csv')
    X = df.drop(columns=['Id', 'Species'])
    y = df['Species']
    y = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    return X, y

def train_test_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def Bagging(X_train, y_train, B):
    s=[]
    for i in range (B):
        sh=resample(X_train, y_train, replace=True)
        s.append(sh)
    for data in s:
        models.fit(data)
    return models

def predict(model, X_test):
    prediction = model.predict(X_test)
    return prediction

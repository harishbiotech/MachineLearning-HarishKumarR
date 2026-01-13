import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR


def load_data():
    X, y = fetch_california_housing(return_X_y=True)
    print(X.shape)
    print(y.shape)
    return X, y
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99998)
    return X_train, X_test, y_train, y_test
def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
def predict(model, X_test, y_test):
    y_predict = model.predict(X_test)
    score = r2_score(y_test, y_predict)
    return score

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    score = predict(model, X_test_scaled, y_test)
    print("r2 score",score)





if __name__ == "__main__":
    main()
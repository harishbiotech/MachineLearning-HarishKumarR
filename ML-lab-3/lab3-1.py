# from pyexpat import model
#
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# from sklearn.linear_model import LinearRegression
#
# def load_data():
#     df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
#     x = df.drop(columns=['disease_score'])
#     y = df['disease_score']
#     print(x)
#     print(y)
#     return x,y
# def split_data(x, y):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=999)
#     return x_train, x_test, y_train, y_test
# def scale_data(x_train, x_test):
#     x_train_scaler = StandardScaler().fit(x_train)
#     x_test_scaler = StandardScaler().fit(x_test)
#     return x_train_scaler, x_test_scaler
# def model_training(x_train_scaler, x_test_scaler, y_train, y_test):
#     model = LinearRegression()
#     model.fit(x_train, y_train)
#     return model
# def prediction(model, x_test, y_test):
#     y_predict = model.predict(x_test)
#     r2 = r2_score(y_test, y_predict)
#     print("r2 score is",r2)
# def main():
#     X, y = load_data()
#     x_train, x_test, y_train, y_test = split_data(X, y)
#     x_train, x_test = scale_data(x_train, x_test)
#     model = model_training(x_train, y_train)
#     prediction(model, x_test, y_test)
#
# if __name__ == "__main__":
#     main()
#
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


def load_data():
    df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X = df.drop(columns=['disease_score','disease_score_fluct'])
    y = df['disease_score']
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=9999)


def scale_data(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def model_training(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def prediction(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("r2 score:", r2_score(y_test, y_pred))


def main():
    X, y = load_data()
    x_train, x_test, y_train, y_test = split_data(X, y)
    x_train, x_test = scale_data(x_train, x_test)
    model = model_training(x_train, y_train)
    prediction(model, x_test, y_test)


if __name__ == "__main__":
    main()

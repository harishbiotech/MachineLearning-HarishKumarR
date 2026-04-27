from XGboost import XGBRegressor
import pandas as pd
from XGboost import XGBClassidier
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data_reg():
    df = pd.read_csv('../ML-lab-15/Boston.csv')
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    return X, y

def spilt_data(X, y):
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size = 0.2, random_state = 0)
    return X_train_reg, X_test_reg, y_train_reg, y_test_reg

def model_regression(X_train_reg, X_test_reg, y_train_reg, y_test_reg):
    model = XGBRegressor()
    model.fit(X_train_reg, y_train_reg)
    return model

def prediction_regression(X_test_reg, y_test_reg):
    prediction_reg = model.predict(X_test_reg)
    return prediction_reg

def r2(y_test, prediction_reg):
    r2 = r2_score(y_test, prediction_reg)
    print("R-squared:",r2)

def load_data_cls():
    


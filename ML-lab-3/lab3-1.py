import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR

def eda(X_df, X_df_scaled):
    pass

def load_data():
    X, y = fetch_california_housing(return_X_y=True)
    print(X.shape)
    print(y.shape)
    return X, y


X, y = load_data()



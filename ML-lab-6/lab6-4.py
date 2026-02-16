#Use validation set to do feature and model selection.
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

import pandas as pd

df = pd.read_csv('data_kfold.csv')
X=df.drop(columns=[df.columns[0], df.columns[1], df.columns[32]])
y=df['diagnosis'].map({'M': 1, 'B': 0})


def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    return X_train, X_test, y_train, y_test

def validation_data(X_train,y_train):
    X_train_train,X_train_validation,y_train_train,y_train_validation = train_test_split(X_train,y_train, test_size = 0.3, random_state = 0)
    return X_train_train,X_train_validation,y_train_train,y_train_validation

def model_linaer_regression(X_train_train,y_train_train):
    model_linear = LinearRegression()
    model_linear.fit(X_train_train,y_train_train)
    return model_linear
def prediction_linear(model_linear,X_test_validation,y_test_validation):
    prediction = model_linear.predict(X_test_validation)
    r2 = r2_score(y_test_validation,prediction)
    return r2
def model_logistic_regression(X_train_train,y_train_train):
    model_logistic = LogisticRegression(max_iter=500)
    model_logistic.fit(X_train_train,y_train_train)
    return model_logistic
def prediction_logistic(model_logistic,X_test_validation,y_test_validation):
    prediction = model_logistic.predict(X_test_validation)
    accuracy = accuracy_score(y_test_validation,prediction)
    return accuracy
def main():
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_train,X_train_validation,y_train_train,y_train_validation= validation_data(X_train,y_train)
    model_linear = model_linaer_regression(X_train_train,y_train_train)
    model_logistic = model_logistic_regression(X_train_train,y_train_train)
    r2 = prediction_linear(model_linear, X_train_validation, y_train_validation)
    accuracy = prediction_logistic(model_logistic, X_train_validation, y_train_validation)

    print("r2_score:", r2)
    print("accuracy_score:", accuracy)

if __name__ == "__main__":
    main()
# Implement a regression decision tree algorithm using scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def loaddata():
    df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X = df.drop(columns=['disease_score', 'disease_score_fluct'])
    y = df['disease_score']
    return X, y


def split(X, y):
    Xtrain, Xtest, ytrain, ytest = train_test_split( X, y, test_size=0.2, random_state=0)
    return Xtrain, Xtest, ytrain, ytest


def scale(Xtrain, Xtest):
    scaler = StandardScaler()
    Xtrain_scale = scaler.fit_transform(Xtrain)
    Xtest_scale = scaler.transform(Xtest)
    return Xtrain_scale, Xtest_scale


def model(Xtrain_scale, ytrain):
    clf = DecisionTreeRegressor(criterion='squared_error', random_state=0)
    clf.fit(Xtrain_scale, ytrain)
    return clf


def prediction(clf, Xtest_scale):
    ypred = clf.predict(Xtest_scale)
    return ypred


def evaluate(ytest, ypred):
    print(f"MAE  : {mean_absolute_error(ytest, ypred):.3f}")
    print(f"MSE  : {mean_squared_error(ytest, ypred):.3f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(ytest, ypred)):.3f}")
    print(f"R²   : {r2_score(ytest, ypred):.3f}")


def visualize_tree(clf, feature_names):
    plt.figure(figsize=(14,8))
    plot_tree(clf, feature_names=feature_names, filled=True)
    plt.show()


def main():
    X, y = loaddata()

    Xtrain, Xtest, ytrain, ytest = split(X, y)

    Xtrain_scale, Xtest_scale = scale(Xtrain, Xtest)

    clf = model(Xtrain_scale, ytrain)

    ypred = prediction(clf, Xtest_scale)

    evaluate(ytest, ypred)

    visualize_tree(clf, X.columns)


if __name__ == "__main__":
    main()
#Implement a regression decision tree algorithm using scikit-learn for the simulated dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

def loaddata():
    df = pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    X = df.drop(columns = ['disease_score','disease_score_fluct'])
    y = df['disease_score']
    return X, y

def split(X,y):
    Xtrain,Xtest,ytrain,ytest=train_test_split(test_size=0.2,random_state=0)
    return Xtrain,Xtest,ytrain,ytest

def scale(Xtrain,Xtest):
    scaler = StandardScaler()
    Xtrain_scale= scaler.fit_transform(Xtrain)
    Xtest_scale= scaler.transform(Xtest)
    return Xtrain_scale,Xtest_scale

def model(Xtrain_scale,Xtest_scale,y_train,y_test):
    clf = DecisionTreeClassifier(criterion='squared_error',random_state=0)
    clf.fit(Xtrain_scale,y_train)
    return clf

def prediction(clf,Xtest_scale,ytest):
    predict= clf.predict(Xtest_scale)
    return predict

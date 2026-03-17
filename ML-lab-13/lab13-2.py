# from sklearn.utils import resample
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import pandas as pd
#
# def load_data():
#     df = pd.read_csv('Iris.csv')
#     X = df.drop(columns=['Id', 'Species'])
#     y = df['Species']
#     y = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
#     return X, y
#
# def train_test_split(X,y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     return X_train, X_test, y_train, y_test
#
# def Bagging(X_train, y_train, B):
#     s=[]
#     for i in range (B):
#         sh=resample(X_train, y_train, replace=True)
#         s.append(sh)
#     for data in s:
#         models.fit(data)
#     return models
#
# def predict(model, X_test):
#     prediction = model.predict(X_test)
#     return prediction
# def score_model(prediction, y_test):
#     accuracy = accuracy_score(y_test, prediction)
#     return accuracy
#
# def main():
#     X,y = load_data()
#     X_train, X_test, y_train, y_test = train_test_split()
#     models = Bagging(X_train, y_train, B=10)
#     prediction = predict(models, X_test)
#     accuracy = score_model(prediction, y_test)
#     print(accuracy)
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


def load_data():
    df = pd.read_csv('Iris.csv')
    X = df.drop(columns=['Id', 'Species'])
    y = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    return X, y


def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def Bagging(X_train, y_train, B):

    models = []

    for i in range(B):

        X_sample, y_sample = resample(X_train, y_train, replace=True)

        model = DecisionTreeClassifier()

        model.fit(X_sample, y_sample)

        models.append(model)

    return models


def predict(models, X_test):

    predictions = []

    for model in models:
        predictions.append(model.predict(X_test))

    predictions = np.array(predictions)

    final_prediction = []

    for i in range(len(X_test)):
        final_prediction.append(np.bincount(predictions[:, i]).argmax())

    return np.array(final_prediction)


def score_model(prediction, y_test):
    accuracy = accuracy_score(y_test, prediction)
    return accuracy


def main():

    X,y = load_data()

    X_train, X_test, y_train, y_test = split_data(X,y)

    models = Bagging(X_train, y_train, B=10)

    prediction = predict(models, X_test)

    accuracy = score_model(prediction, y_test)

    print("Accuracy:",accuracy)


if __name__ == "__main__":
    main()
#Use validation set to do feature and model selection.
# Use validation set to do feature and model selection.

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load dataset
df = pd.read_csv('data_kfold.csv')

X = df.drop(columns=[df.columns[0], df.columns[1], df.columns[32]])
y = df['diagnosis'].map({'M': 1, 'B': 0})


# ---------------- Split Functions ---------------- #

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=0)


def validation_data(X_train, y_train):
    return train_test_split(X_train, y_train, test_size=0.3, random_state=0)


# ---------------- Scaling Function ---------------- #

def scale_data(X_train_train, X_train_validation, X_test):
    scaler = StandardScaler()

    X_train_train_scaled = scaler.fit_transform(X_train_train)

    X_train_validation_scaled = scaler.transform(X_train_validation)
    X_test_scaled = scaler.transform(X_test)

    return X_train_train_scaled, X_train_validation_scaled, X_test_scaled


# ---------------- Models ---------------- #

def model_tree_classifier(X_train_train, y_train_train):
    model_tree = DecisionTreeClassifier()
    model_tree.fit(X_train_train, y_train_train)
    return model_tree


def prediction_tree_classifier(model_tree, X_validation, y_validation):
    prediction = model_tree.predict(X_validation)
    return prediction


def model_logistic_regression(X_train_train, y_train_train):
    model_logistic = LogisticRegression(solver='saga',l1_ratio=1.0,C=0.5,max_iter=5000)
    model_logistic.fit(X_train_train, y_train_train)
    return model_logistic


def prediction_logistic(model_logistic, X_validation, y_validation):
    prediction = model_logistic.predict(X_validation)
    return prediction


def main():

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_train, X_train_validation, y_train_train, y_train_validation = validation_data(X_train, y_train)

    X_train_train_scaled, X_train_validation_scaled, X_test_scaled = scale_data(X_train_train, X_train_validation, X_test)

    model_linear = model_tree_classifier(X_train_train_scaled, y_train_train)
    model_logistic = model_logistic_regression(X_train_train_scaled, y_train_train)

    predict_tree = prediction_tree_classifier(model_linear, X_train_validation_scaled, y_train_validation)
    predict_logistic = prediction_logistic(model_logistic, X_train_validation_scaled, y_train_validation)

    accuracy_tree = accuracy_score(y_train_validation, predict_tree)
    accuracy_logistic = accuracy_score(y_train_validation, predict_logistic)

    print("Validation acuuracy_score (Tree):", accuracy_tree)
    print("Validation accuracy_score (Logistic):", accuracy_logistic)


if __name__ == "__main__":
    main()
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import  LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import r2_score
# from sklearn.preprocessing import StandardScaler
#
# import pandas as pd
#
# df = pd.read_csv('data_kfold.csv')
# X=df.drop(columns=[df.columns[0], df.columns[1], df.columns[32]])
# y=df['diagnosis'].map({'M': 1, 'B': 0})
#
#
# def split_data(X,y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
#     return X_train, X_test, y_train, y_test
#
# def validation_data(X_train,y_train):
#     X_train_train,X_train_validation,y_train_train,y_train_validation = train_test_split(X_train,y_train, test_size = 0.3, random_state = 0)
#     return X_train_train,X_train_validation,y_train_train,y_train_validation
# def scaled_data(X_train_train,X_train_validation,y_train_train,y_train_validation):
#     scaler = StandardScaler()
#     X_train_trainscaler.fit_transform(X_train_train)
#     scaler.transform(X_train_validation)
#     scaler.transform(y_train_validation)
#     return
#
# def model_linaer_regression(X_train_train,y_train_train):
#     model_linear = LinearRegression()
#     model_linear.fit(X_train_train,y_train_train)
#     return model_linear
# def prediction_linear(model_linear,X_test_validation,y_test_validation):
#     prediction = model_linear.predict(X_test_validation)
#     r2 = r2_score(y_test_validation,prediction)
#     return r2
# def model_logistic_regression(X_train_train,y_train_train):
#     model_logistic = LogisticRegression(max_iter=500)
#     model_logistic.fit(X_train_train,y_train_train)
#     return model_logistic
# def prediction_logistic(model_logistic,X_test_validation,y_test_validation):
#     prediction = model_logistic.predict(X_test_validation)
#     accuracy = accuracy_score(y_test_validation,prediction)
#     return accuracy
# def main():
#     X_train, X_test, y_train, y_test = split_data(X,y)
#     X_train_train,X_train_validation,y_train_train,y_train_validation= validation_data(X_train,y_train)
#     model_linear = model_linaer_regression(X_train_train,y_train_train)
#     model_logistic = model_logistic_regression(X_train_train,y_train_train)
#     r2 = prediction_linear(model_linear, X_train_validation, y_train_validation)
#     accuracy = prediction_logistic(model_logistic, X_train_validation, y_train_validation)
#
#     print("r2_score:", r2)
#     print("accuracy_score:", accuracy)
#
# if __name__ == "__main__":
#     main()
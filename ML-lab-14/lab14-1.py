import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_csv("Iris.csv")

    X = df.drop(columns=["Id","Species"])
    y = df["Species"]

    y = y.map({
        "Iris-setosa":0,
        "Iris-versicolor":1,
        "Iris-virginica":2
    })

    return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    return X_train, X_test, y_train, y_test

def model(X_train, y_train):
    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5),n_estimators=100,learning_rate=0.1,random_state=0)
    model.fit(X_train,y_train)
    return model

def predict(model, X_test, y_test):
    predictions = model.predict(X_test)
    return predictions

def accuracy(predictions, y_test):
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    models = model(X_train, y_train)
    predictionss = predict(models,X_test,y_test)
    accuracyy = accuracy(predictionss, y_test)
    print("Accuracy score is :",accuracyy)

if __name__ == "__main__":
    main()
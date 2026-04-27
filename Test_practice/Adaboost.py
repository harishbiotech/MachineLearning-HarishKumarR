import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_csv('../ML-lab-14/Iris.csv')
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
    return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
    return X_train, X_test, y_train, y_test

def model(X_train, y_train):
    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5),n_estimators = 100, learning_rate = 0.1, random_state = 10)
    model.fit(X_train,y_train)
    return model

def predict(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred

def accuracy(y_test,y_pred):
    print(f'accuracy: {accuracy_score(y_test,y_pred)*100}%')

def main():
    X,y = load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    Model = model(X_train, y_train)
    y_pred = predict(Model,X_test)
    acc=accuracy(y_test,y_pred)


if __name__ == '__main__':
    main()
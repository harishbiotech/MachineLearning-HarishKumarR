import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_csv('sonar.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_np=X.values
    y_np=y.values
    return X_np, y_np

def split_data(X_np, y_np):
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size = 0.2, random_state = 999)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def model(X_train_scaled, y_train):
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train_scaled, y_train)
    return clf

def predict(clf, X_test_scaled, y_test):
    prediction = clf.predict(X_test_scaled)
    return prediction

def evaluate(prediction, y_test):
    acc=accuracy_score(y_test, prediction)
    return acc

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    clf = model(X_train_scaled, y_train)
    prediction = predict(clf, X_test_scaled, y_test)
    acc = evaluate(prediction, y_test)
    print('Accuracy score:', acc)

if __name__ == '__main__':
    main()



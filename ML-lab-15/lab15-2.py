from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    df = pd.read_csv('Weekly.csv')
    X=df.drop(columns=['Direction'])
    y=df['Direction'].map({'Down':-1,'Up':1})
    return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    return X_train, X_test, y_train, y_test

def model(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    print("Accuracy:", accuracy)

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    modell = model(X_train, y_train)
    prediction = predict(modell, X_test, y_test)

if __name__ == "__main__":
    main()
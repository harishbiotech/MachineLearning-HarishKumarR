from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd

def load_data():
    df = pd.read_csv('../ML-lab-15/Weekly.csv')
    X=df.iloc[:,1:-1]
    y=df.iloc[:,-1].map({'Down':0,'Up':1})
    return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
    return X_train, X_test, y_train, y_test

def model_cls(X_train, y_train):
    model = XGBClassifier(n_estimators=100,learning_rate=0.1)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test, y_test):
    prediction = model.predict(X_test)
    print(f'accuracy: {accuracy_score(y_test, prediction)}')

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    modell = model_cls(X_train, y_train)
    prediction = predict(modell, X_test, y_test)

if __name__ == '__main__':
    main()

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd

def load_data():
    df = pd.read_csv('../ML-lab-15/Boston.csv')
    X=df.iloc[:,1:-1]
    y=df.iloc[:,-1]
    return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
    return X_train, X_test, y_train, y_test

def model_regressor(X_train, y_train ):
    model_reg= XGBRegressor(n_estimators=100,learning_rate=0.1)
    model_reg.fit(X_train,y_train)
    return model_reg

def model_classifier(X_train, y_train ):
    model_cls= XGBClassifier(n_estimators=100,learning_rate=0.1)
    model_cls.fit(X_train,y_train)
    return model_cls

def predict(model_reg, X_test, y_test ):
    prediction_reg = model_reg.predict(X_test)
    print(f'r2_score : {r2_score(y_test, prediction_reg)}')


def main ():
    X,y=load_data()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
    model_reg=model_regressor(X_train, y_train)
    prediction=predict(model_reg, X_test, y_test)

if __name__=="__main__":
    main()




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

def load_data():
    df=pd.read_csv('../ML-lab-15/Boston.csv')
    X=df.drop(columns=['medv'])
    y=df['medv']
    return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
    return X_train, X_test, y_train, y_test

def standardize(X_train,X_test):
    scaler = StandardScaler()
    X_std_train = scaler.fit_transform(X_train)
    X_std_test = scaler.transform(X_test)
    return X_std_train, X_std_test

def model(X_std_train, y_train):
    model = GradientBoostingRegressor()
    model.fit(X_std_train,y_train)
    return model

def predict(model,X_std_test):
    prediction = model.predict(X_std_test)
    return prediction

def mse(y_test,y_pred):
    m= mean_squared_error(y_test,y_pred)
    print(m)
    r2 = r2_score(y_test,y_pred)
    print(r2)

def main():
    x,y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(x,y)
    X_std_train, X_std_test = standardize(X_train,X_test)
    Model=model(X_std_train,y_train)
    prediction = predict(Model,X_std_test)
    mse(y_test,prediction)


if __name__ == '__main__':
    main()
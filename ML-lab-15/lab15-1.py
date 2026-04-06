from pandas.core.interchange import column
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data():
    df=pd.read_csv('Boston.csv')
    X=df.drop(columns=['medv'])
    y=df['medv']
    return X,y

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_std_train= scaler.fit_transform(X_train)
    X_std_test= scaler.transform(X_test)
    return X_std_train, X_std_test

def model(X_train, X_test, y_train, y_test):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=10)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test, y_test):
    prediction = model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    print("MSE: ", mse)

def main():
    X,y = load_data()
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_std_train, X_std_test = standardize_data(X_train, X_test)
    modell= model(X_std_train, X_std_test, y_train, y_test)
    prediction = predict(modell, X_std_test, y_test)

if __name__ == "__main__":
    main()





import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def load_data():
    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X=df.drop(columns=["disease_score_fluct","disease_score"])
    y=df['disease_score_fluct']
    return X, y
def split_data(X, y):
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train,X_test,y_train,y_test
def scale_data(X_test, X_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_test, X_train
def model_training(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
def prediction(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return y_pred, r2
def main():
     load_data()
     X, y = load_data()
     X_train, X_test, y_train, y_test = split_data(X, y)
     model = model_training(X_train, X_test, y_train, y_test)
     y_pred, r2 = prediction(model, X_test, y_test)
     print("r2 :",r2)
     print(y_pred)
     # print(y_pred)
if __name__ == "__main__":
    main()

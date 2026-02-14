#Implement logistic regression using scikit-learn for the breast cancer dataset - https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data ===============
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ---------------- LOAD DATA ----------------
def load_data():
    df = pd.read_csv('data.csv')
    X = df.drop(columns=[df.columns[0], df.columns[1], df.columns[32]])
    y = df['diagnosis']

    # Convert target to numeric (M=1, B=0)
    y = y.map({'M': 1, 'B': 0})

    return X, y

# ---------------- SPLIT DATA ----------------
def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=0)

# ---------------- STANDARDIZATION ----------------
def standardize(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# ---------------- MODEL TRAINING ----------------
def model_training(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_scaled, y_train)
    return model

# ---------------- PREDICTION ----------------
def predict(model, X_test_scaled, y_test):
    prediction = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, prediction)
    return acc, prediction

# ---------------- MAIN ----------------
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled = standardize(X_train, X_test)

    model = model_training(X_train_scaled, y_train)
    acc, prediction = predict(model, X_test_scaled, y_test)

    print("Accuracy =", acc)
    print("Predictions =", prediction)

if __name__ == "__main__":
    main()






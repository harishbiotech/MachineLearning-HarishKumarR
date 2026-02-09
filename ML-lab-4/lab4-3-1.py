# ================= Admission Prediction using Linear Regression =================

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# --------------------- Load & Clean Data ---------------------
def load_data():
    df = pd.read_csv("Admission_Predict.csv")

    # Clean column names (VERY IMPORTANT)
    df.columns = df.columns.str.strip().str.lower()

    # Separate features and target
    X = df.drop(columns=["chance of admit"])
    y = df["chance of admit"]

    return X, y


# --------------------- Train-Test Split ---------------------
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=0)


# --------------------- Feature Scaling ---------------------
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# --------------------- Model Training ---------------------
def model_training(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# --------------------- Prediction & Evaluation ---------------------
def prediction(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return y_pred, r2


# --------------------- Main Function ---------------------
def main():
    # Load data
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Train model
    model = model_training(X_train_scaled, y_train)

    # Predict
    y_pred, r2 = prediction(model, X_test_scaled, y_test)

    # Output
    print("R² Score:", r2)



# --------------------- Run Program ---------------------
if __name__ == "__main__":
    main()

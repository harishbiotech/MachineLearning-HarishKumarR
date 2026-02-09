#read csv file
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

df = pd.read_csv('housing.csv')
df = df.fillna(df.mean(numeric_only=True))

# X and y
X_train = df.iloc[:16512, :8]
y_train = df.iloc[:16512, 8:9]
X_test = df.iloc[16513:, :8]
y_test = df.iloc[16513:, 8:9]

X_np_train = X_train.values
X_np_test = X_test.values

y_np_train = y_train.values.reshape(-1, 1)
y_np_test = y_test.values.reshape(-1, 1)

# ---------------- SCALING (FIX: scale BEFORE bias) ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_np_train)
X_test_scaled = scaler.transform(X_np_test)

# ---------------- ADD BIAS AFTER SCALING (FIX) ----------------
m = X_train_scaled.shape[0]
X_train_scaled = np.concatenate((np.ones((m,1)), X_train_scaled), axis=1)

n = X_test_scaled.shape[0]
X_test_scaled = np.concatenate((np.ones((n,1)), X_test_scaled), axis=1)

print("X train scaled =", X_train_scaled)
print("X test scaled =", X_test_scaled)

# ---------------- INITIALIZE THETA ----------------
theta = np.zeros((X_train_scaled.shape[1]))
theta_matrix = theta.reshape(9,1)
print("Theta", theta_matrix)

# ---------------- HYPOTHESIS ----------------
def hypothesis(X_train_scaled, theta_matrix):
    return np.dot(X_train_scaled, theta_matrix)

h = hypothesis(X_train_scaled, theta_matrix)

# -------------------cost----------------------
def cost(h, y_np):
    for i in range(len(y_np_train)):
        s = 0
        s += ((h[i][0] - y_np[i][0]) ** 2)
    return s*1/2

# ---------------- GRADIENT DESCENT ----------------
alpha = 0.00001 # FIX: smaller learning rate

def find_theta(X_train_scaled, y_np_train, h, alpha, theta_matrix):
    thet = []

    for i in range(len(X_train_scaled[0])):
        s = 0
        for j in range(len(y_np_train)):
            s += (h[j][0] - y_np_train[j][0]) * X_train_scaled[j][i]

        # FIX: divide by m
        thet.append(theta_matrix[i][0] - (alpha * s))

    theta = np.array(thet)
    return theta.reshape(-1,1)

# ---------------- TRAIN LOOP ----------------
f = find_theta(X_train_scaled, y_np_train, h, alpha, theta_matrix)
print("find_theta =", f)

for i in range(250):
    Theta = f
    hypothes = hypothesis(X_train_scaled, Theta)
    prev_cost = cost(hypothes, y_np_train)
    print(i, "cost", prev_cost)
    theta_update = find_theta(X_train_scaled, y_np_train, hypothes, alpha, Theta)

    print(i, "theta_updated", theta_update)
    print(i, "hypothesis sample", hypothes)

    Theta = theta_update
theta_for_prediction=theta_update
print("theta_to_test",theta_for_prediction)
print("-"*30,"Testing the model","-"*90)
Prediction_testing=hypothesis(X_test_scaled,theta_for_prediction)
print("Predicted y values = ",Prediction_testing)
r_square_value=r2_score(y_np_test,Prediction_testing)
print("r_square_value = ",r_square_value)


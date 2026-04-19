import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc


df = pd.read_csv("Heart.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)
df["AHD"] = df["AHD"].map({"Yes": 1, "No": 0})
X = df.drop("AHD", axis=1)
y = df["AHD"]
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)

model = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=100))
])

model.fit(X_train, y_train)


y_prob = model.predict_proba(X_test)[:, 1]
def evaluate(threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) != 0 else 0

    print("\nThreshold =", threshold)
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy   :", round(accuracy, 4))
    print("Precision  :", round(precision, 4))
    print("Sensitivity:", round(sensitivity, 4))
    print("Specificity:", round(specificity, 4))
    print("F1-score   :", round(f1, 4))


for t in [0.3, 0.5, 0.7]:
    evaluate(t)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="AUC = " % roc_auc)
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()
print("\nAUC Score =", round(roc_auc, 4))
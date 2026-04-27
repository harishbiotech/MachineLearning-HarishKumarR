import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -----------------------------
# Generate Data
# -----------------------------
X_train, y_train = make_moons(n_samples=100, noise=0.3, random_state=42)
X_test, y_test = make_moons(n_samples=100, noise=0.3, random_state=24)

df_train = pd.DataFrame(X_train, columns=['x1', 'x2'])
df_train['y'] = y_train

# -----------------------------
# Plot function (given)
# -----------------------------
def plot_clf(model, df, grid_range, show_contours=False):
    x1 = grid_range
    x2 = grid_range
    xx1, xx2 = np.meshgrid(x1, x2)
    Xgrid = np.c_[xx1.ravel(), xx2.ravel()]

    decision_boundary = model.predict(Xgrid)
    decision_boundary = decision_boundary.reshape(xx1.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx1, xx2, decision_boundary, alpha=0.3)

    sns.scatterplot(x='x1', y='x2', hue='y', data=df)
    plt.title("Decision Boundary")
    plt.show()

# -----------------------------
# 1. Linear SVM
# -----------------------------
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

# -----------------------------
# 2. Polynomial SVM
# -----------------------------
poly_svm = SVC(kernel='poly', degree=3)
poly_svm.fit(X_train, y_train)

# -----------------------------
# 3. RBF SVM
# -----------------------------
rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)

# -----------------------------
# Accuracy
# -----------------------------
models = {
    "Linear": linear_svm,
    "Polynomial": poly_svm,
    "RBF": rbf_svm
}

for name, model in models.items():
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} SVM -> Train: {train_acc:.3f}, Test: {test_acc:.3f}")

# -----------------------------
# Plot decision boundaries
# -----------------------------
grid = np.linspace(-2, 3, 100)

print("\nLinear SVM Plot")
plot_clf(linear_svm, df_train, grid)

print("Polynomial SVM Plot")
plot_clf(poly_svm, df_train, grid)

print("RBF SVM Plot")
plot_clf(rbf_svm, df_train, grid)
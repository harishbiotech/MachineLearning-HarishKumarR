from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

# select class 1 & 2
mask = y != 0
X = X[mask]
y = y[mask]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y
)

model = SVC(kernel='rbf')
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))


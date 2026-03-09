from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('diabetes.csv')
X=df.drop(columns='Outcome')
y=df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model=BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=0
)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("Accuracy", accuracy_score(y_test,y_pred))
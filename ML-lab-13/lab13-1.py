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

print('-'*100)

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
model_regressor=BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=10,
    random_state=0
)

model_regressor.fit(X_train,y_train)

y_pred_reg=model_regressor.predict(X_test)

print('r2_score', r2_score(y_test,y_pred_reg))
print('-'*100)
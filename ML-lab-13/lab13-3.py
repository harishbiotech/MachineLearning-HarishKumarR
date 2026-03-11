from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,r2_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df = pd.read_csv('diabetes.csv')
X=df.drop(columns='Outcome')
y=df['Outcome']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

model_regression=RandomForestRegressor(n_estimators=42,random_state=0)
model_classification=RandomForestClassifier(n_estimators=42,random_state=0)

model_regression.fit(X_train,y_train)
model_classification.fit(X_train,y_train)

predict_classification=model_classification.predict(X_test)
predict_regression=model_regression.predict(X_test)

print('Diabetes')
print('Accuracy',accuracy_score(y_test,predict_classification))
print('r2_score',r2_score(y_test,predict_regression))
print('-'*60)

df = pd.read_csv('Iris.csv')
X = df.drop(columns=['Id', 'Species'])
y=df['Species']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

encoder=OneHotEncoder()

y_train_coded = encoder.fit_transform(y_train.values.reshape(-1,1)).toarray()
y_test_coded = encoder.transform(y_test.values.reshape(-1,1)).toarray()

model_regression=RandomForestRegressor(n_estimators=100,random_state=0)
model_classification=RandomForestClassifier(n_estimators=42,random_state=0)

model_regression.fit(X_train,y_train_coded)
model_classification.fit(X_train,y_train)

predict_classification=model_classification.predict(X_test)
predict_regression=model_regression.predict(X_test)

print('accuracy_score',accuracy_score(y_test,predict_classification))
print('r2score',r2_score(y_test_coded,predict_regression))


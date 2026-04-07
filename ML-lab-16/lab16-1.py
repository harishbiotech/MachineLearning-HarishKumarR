from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score

def load_data():
    df = pd.read_csv('../ML-lab-15/Boston.csv')
    X=df.drop('medv',axis=1)
    y=df['medv']
    return X,y

X,y=load_data()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
tree=[]
B=10
for i in range(B):
    model = DecisionTreeRegressor(max_depth=10)
    model.fit(X_train,y_train)
    tree.append(model)

score=[]
for model in tree:
    prediction=model.predict(X_test)
    score.append(r2_score(y_test,prediction))

final_prediction=sum(score)/len(score)
print(final_prediction)

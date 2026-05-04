import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv('Heart.csv')
X=df.drop(
    columns=['AHD']
)
Y=df['AHD'].map({'Yes':1,'No':0})

encode=OneHotEncoder()
X_encoded = encode.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_encoded,Y,test_size=0.2,random_state=100)

model = LogisticRegression()
model.fit(X_train,y_train)

scores= model.predict_proba(X_test)
# print(scores)
scores_1=scores[:,0]
scores_2=scores[:,1]
print(scores_1)
print(scores_2)

y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test,y_pred)
print('Confusion Matrix: ',conf_mat)

TP = conf_mat[0,0]
TN = conf_mat[1,1]
FP = conf_mat[0,1]
FN = conf_mat[1,0]

print(f"TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}")

Accuracy=TP+TN/(TP+TN+FP+FN)
Precision=TP/(TP+FP)
Recall=TP/(TP+FN)
specificity=TN/(TN+FP)
F1_Score = 2*((TP/(TP+FP)*(TP/(TP+FN)))/((TP/(TP+FP))+(TP/(TP+FN))))

print("Accuracy:",Accuracy)
print("Precision:",Precision)
print("Recall:",Recall)
print("Specificity:",specificity)
print("F1_Score:",F1_Score)

auc=roc_auc_score(y_test,scores_2)
print(f'AUC:{auc}')

fpr,tpr,thresholds = roc_curve(y_test,scores_2)
#plt.label('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fpr,tpr,label='ROC Curve')
plt.grid()
plt.show()
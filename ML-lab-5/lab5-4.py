#Implement logistic regression using scikit-learn for the breast cancer dataset - https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data ===============
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
def load_data():


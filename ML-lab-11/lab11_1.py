import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math

def load_data():
    df = pd.read_csv("Iris.csv")
    X = df.drop(columns=['Id', 'Species'])
    y = df['Species'].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
    return X, y


def root_entropy(y):
    proportions = y.value_counts(normalize=True).tolist()
    entropy = sum(-p * math.log(p, 2) for p in proportions if p > 0)
    return entropy

def split_points(X):
    splits = {}
    for col in X.columns:
        sorted_vals = X[col].sort_values().unique()
        midpoints = [(sorted_vals[i] + sorted_vals[i+1]) / 2 for i in range(len(sorted_vals)-1)]
        splits[col] = midpoints
    return splits

def information_gain(X_col, y, threshold):
    left_y  = y[X_col <= threshold]
    right_y = y[X_col > threshold]

    n = len(y)
    n_left  = len(left_y)
    n_right = len(right_y)

    if n_left == 0 or n_right == 0:
        return 0

    ig = root_entropy(y) - (n_left/n) * root_entropy(left_y) - (n_right/n) * root_entropy(right_y)
    return ig


def best_split(X, y):
    splits = split_points(X)
    best_ig = 0
    best_col = None
    best_threshold = None

    for col, thresholds in splits.items():
        for threshold in thresholds:
            ig = information_gain(X[col], y, threshold)
            if ig > best_ig:
                best_ig = ig
                best_col = col
                best_threshold = threshold

    return best_col, best_threshold, best_ig

def majority_class(y):
    # Returns the most common class in y
    return y.value_counts().idxmax()


def build_tree(X, y, depth=0, max_depth=5, min_samples=5):
    #  STOP condition 1: all samples are same class
    print(type(y))
    if len(y.unique()) == 1:
        return {"leaf": True, "class": y.iloc[0]}

    #  STOP condition 2: max depth reached
    if depth >= max_depth:
        return {"leaf": True, "class": majority_class(y)}

    # STOP condition 3: too few samples
    if len(y) < min_samples:
        return {"leaf": True, "class": majority_class(y)}

    #  STEP 1: Find the best split
    col, threshold, ig = best_split(X, y)

    #  STOP condition 4: no useful split found
    if ig == 0:
        return {"leaf": True, "class": majority_class(y)}

    #  STEP 2: Split data into left and right
    left_mask = X[col] <= threshold
    right_mask = X[col] > threshold

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]

    # STEP 3: Recurse on left and right
    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth, min_samples)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth, min_samples)

    # Return the decision node
    return {
        "leaf": False,
        "col": col,
        "threshold": threshold,
        "left": left_subtree,  # X[col] <= threshold goes left
        "right": right_subtree  # X[col] >  threshold goes right
    }


def predict_one(tree, row):
    # row is a single sample (pandas Series)

    if tree["leaf"]:
        return tree["class"]  # we reached a leaf, return the class

    # Go left or right based on the split condition
    if row[tree["col"]] <= tree["threshold"]:
        return predict_one(tree["left"], row)
    else:
        return predict_one(tree["right"], row)


def predict(tree, X):
    return X.apply(lambda row: predict_one(tree, row), axis=1)

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the tree
tree = build_tree(X_train, y_train, max_depth=5)

# Predict
y_pred = predict(tree, X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))



























# def load_data():
#     df = pd.read_csv("Iris.csv")
#     X=df.drop(columns=['Id','Species'])
#     y=df['Species'].map({ "Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2})
#     return X,y
#
# def root_entropy(y):
#     probs = y.value_counts(normalize=True)
#     s = 0
#     for p in probs:
#         s += -p * math.log(p, 2)
#     return s
# # y_unique=y.value_counts(normalize=True)
#     # #print(y_unique)
#     # proportions=y_unique.to_list()
#     # #print(proportions)
#     # entropy=[]
#     # s=0
#     # for i in range(len(proportions)):
#     #     s = s + (-proportions[i]) * (math.log(proportions[i], 3))
#     # entropy=s
#     # return entropy
#
# def Information_gain(X,y):
#     X=X.tolist()
#
#     def split(X):
#         for i in range(len(X.columns)):
#             for j in range(len(X.iloc[:, :1])):
#                 a = (X.iloc[:j, :i] + X.iloc[:j + 1, :i]) / 2
#         return a
#
#
# def split_points(X):
#     for i in range(len(X.columns)):
#         for j in range(len(X.iloc[:,:1])):
#             a=(X.iloc[:j,:i]+X.iloc[:j+1,:i])/2
#     return a
#
# def IG_entropy_for_columns(a,X,y):
#     for col in a.columns:
#         for i in range(len(a[col])):
#             #left_node=pd.concat([X[col][X[col]>a[col][i]],y[X[col]>a[col][i]]], axis=1)
#             left_node=y[X[col]>a[col][i]]
#             print(root_entropy(left_node))
#             right_node=X[col][X[col] <= a[col][i]]
#             print(root_entropy(right_node))
#             # print(left_node)

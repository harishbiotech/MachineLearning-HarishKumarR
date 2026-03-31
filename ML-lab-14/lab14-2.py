import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from math import log, exp

def load_data():
    df = pd.read_csv("Iris.csv")
    X = df.drop(columns=["Id", "Species"])
    y = df["Species"]
    y = y.map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
    return X.values, y.values


class AdaBoostClassifier:
    def __init__(self):
        self.__stumps = []
        self.__lambdas = []

    def __get_initial_weights(self, X):
        n = len(X)
        return pd.DataFrame({'weights': [1 / n for _ in range(n)]})

    def __epsilon(self, y_true, y_pred, weights):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        misclassified = list(y_pred != y_true)
        w = np.array(weights['weights'])
        epsilon = np.sum(np.where(y_true == y_pred, 0, w))
        return epsilon, misclassified

    def __get_weights(self, y_true, y_pred, epsilon, weights, misclassified):
        n = len(y_pred)
        epsilon = np.clip(epsilon, 1e-10, 1 - 1e-10)
        lamda = 0.5 * log((1 - epsilon) / epsilon)
        new_weights = [
            weights.iloc[i, 0] * exp(lamda) if misclassified[i]
            else weights.iloc[i, 0] * exp(-lamda)
            for i in range(n)
        ]
        total = np.sum(new_weights)
        weights = pd.DataFrame({'weights': new_weights / total})
        return weights, lamda

    def __stump_train(self, X, y, weights):
        md = DecisionTreeClassifier(max_depth=1)
        md.fit(X, y, sample_weight=weights['weights'])
        self.__stumps.append(md)
        return md

    def __stump_predict(self, X, stump):
        return stump.predict(X)

    def fit(self, X, y, max_iter=50):
        weights = self.__get_initial_weights(X)
        for i in range(max_iter):
            stump = self.__stump_train(X, y, weights)
            y_pred = self.__stump_predict(X, stump)
            epsilon, misclassified = self.__epsilon(y, y_pred, weights)
            if epsilon >= 0.5:
                print(f"Stopping early at iteration {i} — epsilon={epsilon:.4f}")
                self.__stumps.pop()
                break
            weights, lamda = self.__get_weights(y, y_pred, epsilon, weights, misclassified)
            self.__lambdas.append(lamda)

    def predict(self, X):
        stump_preds = np.array([stump.predict(X) for stump in self.__stumps])
        classes = np.unique(stump_preds)
        n_samples = X.shape[0]
        class_scores = np.zeros((n_samples, len(classes)))
        for stump_pred, lam in zip(stump_preds, self.__lambdas):
            for j, cls in enumerate(classes):
                class_scores[:, j] += lam * (stump_pred == cls)
        return classes[np.argmax(class_scores, axis=1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


def main():
    # 1. Load data
    X, y = load_data()

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train
    model = AdaBoostClassifier()
    model.fit(X_train, y_train, max_iter=50)

    # 4. Evaluate
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test,  y_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")


if __name__ == "__main__":
    main()
# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from math import log2
# def load_data():
#
#     df = pd.read_csv("Iris.csv")
#
#     X = df.drop(columns=["Id","Species"])
#     y = df["Species"]
#
#     y = y.map({
#         "Iris-setosa":0,
#         "Iris-versicolor":1,
#         "Iris-virginica":2
#     })
#
#     return X.values,y.values
#
# class AdaBoostClassifier:
#     def __init__(self):
#         self.__stumps = []
#         self.__lambda = []
#
#     def __get_initial_weights(self,X):
#         n = len(X)
#         return pd.DataFrame({'weights':[1/n for i in range(n)]})
#
#     def __epsilon(self,y_true,y_pred,weights):
#         y_true = np.array(y_true)
#         y_pred = np.array(y_pred)
#         misclassified = (y_pred != y_true)
#         misclassified = list(misclassified)
#         w=np.array(weights['weights'])
#         epsilon = np.sum(np.where(y_true == y_pred,0,w))
#         return epsilon,misclassified
#
#     def __get_weights(self,y_true,y_pred,epsilon,weights,misclassified):
#         n=len(y_pred)
#         lamda=0.5*(log2(1-epsilon)-epsilon)
#         weights=[weights.iloc[i,0]*np.exp(lamda) if misclassified[i] == True else weights.iloc[i,0]*np.exp(1-lamda) for i in range(n)]
#         weights=pd.DataFrame({'weights':weights/np.sum(weights)})
#         return weights,lamda
#
#     def __stump_train(self,X,stumps):
#         md=DecisionTreeClassifier(max_depth=1)
#         md.fit(X,y,sample_weight=weights['weights'])
#         self.__stumps.append(md)
#         return md
#
#     def __stump_predict(self,X,stumps):
#         y_pred=stumps.predict(X)
#         return y_pred
#
#     def fit(self,X,y,max_iter=1000):

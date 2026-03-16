# Implement ordinal encoding and one-hot encoding methods in Python from scratch.
import pandas as pd

class ordinal_encoding():
    def __init__(self):
        return
    def fit_transform(self,data):
        self.data = data
        self.categories = {}
        for col in self.data.columns:
            for idx, value in enumerate(sorted((self.data[col]).unique())):
                self.categories[value] = idx



data=pd.DataFrame({'y':['A','B','C','D'],
                   'x':['E','F','G','H'],})
print(ordinal_encoding().fit_transform(data))

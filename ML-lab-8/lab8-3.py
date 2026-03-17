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
        return self.data

class one_hot_encoding():
    def __init__(self):
        return
    def fit_transform(self,data):
        self.data = data
        self.categories = {}
        new_df = self.data.copy()
        for col in self.data.columns:
            self.categories[col] = self.data[col].unique()
        for col in self.data.columns:
            for category in self.categories[col]:
                new_col_name = f"{col}_{category}"
                new_df[new_col_name] = (new_df[col] == category).astype(int)
            new_df.drop(columns=[col], inplace=True)
        return new_df






data=pd.DataFrame({'y':['A','B','C','D'],
                   'x':['E','F','G','H'],})
# print(ordinal_encoding().fit_transform(data))
one_hot_data = pd.DataFrame({'y':['Red','Yellow','Green','Blue','Blue','Red']})
print(one_hot_encoding().fit_transform(one_hot_data))

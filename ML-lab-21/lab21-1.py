import numpy as np
import pandas as pd
from pandas.core.common import random_state
import math as math
import matplotlib.pyplot as plt
#create pandas data frame

X={'X1':[1,1,2,8,8,9],
 'X2':[1,2,2,8,9,8] }
df = pd.DataFrame(X)
mu=df.mean(axis=0)
print(mu)
#Randomly assign samples in two cluster
# Shuffle rows
df_shuffled = df.sample(frac=1, random_state=10)
split_index = int(0.5 * len(df))

c1 = df_shuffled.iloc[:split_index]
c2 = df_shuffled.iloc[split_index:]
print(c1)
print(c2)

c1_avg = c1.mean(axis=0)
c2_avg = c2.mean(axis=0)


def distance(p1, p2):
 return math.sqrt((p1['X1'] - p2['X1']) ** 2 + (p1['X2'] - p2['X2']) ** 2)


df['c1_point'] = df.apply(lambda x: distance(x, c1_avg), axis=1)
df['c2_point'] = df.apply(lambda x: distance(x, c2_avg), axis=1)

df['cluster'] = df[['c1_point', 'c2_point']].idxmin(axis=1)
df_cluster_1 = df[df['cluster'] == 'c1_point'].drop(columns=['c1_point', 'c2_point'])
df_cluster_2 = df[df['cluster'] == 'c2_point'].drop(columns=['c1_point', 'c2_point'])

print(df)
print(df_cluster_1)
print(df_cluster_2)

plt.scatter(df_cluster_1['X1'],df_cluster_1['X2'],label='cluster 1')
plt.scatter(df_cluster_2['X1'],df_cluster_2['X2'],label='cluster 2')
plt.legend()
plt.show()


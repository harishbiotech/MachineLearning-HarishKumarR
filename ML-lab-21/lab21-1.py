import numpy as np
import pandas as pd

#create pandas data frame
X={'X1':[1,1,2,8,8,9],
 'X2':[1,2,2,8,9,8] }
df = pd.DataFrame(X)
mu=df.mean(axis=0)
print(mu)
#Randomly assign smaples in

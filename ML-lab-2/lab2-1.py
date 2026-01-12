X=[
    [1,0,2],
    [0,1,1],
    [2,1,0],
    [1,1,1],
    [0,2,1]
]
X_transpose=[]
for i in range(len(X[0])):
    t=[]
    for j in range(len(X)):
        t.append(X[j][i])
    X_transpose.append(t)
print("X transpose =",X_transpose)
#covariance matrix X_transpose*X
XXt=[]
for i in range(len(X)):
    row=[]
    for j in range(len(X[0])):
        a=0
        for k in range(len(X)):
            a=a+(X[i][j]*X_transpose[j][k])
        row.append(a)
    XXt.append(row)
print(XXt)
means = []
for j in range(len(X[0])):        # columns
    s = 0
    for i in range(len(X)):       # rows
        s += X[i][j]
    means.append(s / len(X))

print("Means =", means)
X_centered = []
for i in range(len(X)):
    row = []
    for j in range(len(X[0])):
        row.append(X[i][j] - means[j])
    X_centered.append(row)

print("Mean centered X =", X_centered)
Xc_transpose = []
for i in range(len(X_centered[0])):
    t = []
    for j in range(len(X_centered)):
        t.append(X_centered[j][i])
    Xc_transpose.append(t)

print("X_centered transpose =", Xc_transpose)
cov_matrix = []

for i in range(len(Xc_transpose)):        # rows (features)
    row = []
    for j in range(len(X_centered[0])):   # columns (features)
        a = 0
        for k in range(len(X_centered)):  # samples
            a += Xc_transpose[i][k] * X_centered[k][j]
        row.append(a)
    cov_matrix.append(row)

for i in range(len(cov_matrix)):
    for j in range(len(cov_matrix[0])):
        cov_matrix[i][j] = cov_matrix[i][j] / (len(X) - 1)

print("Covariance Matrix =", cov_matrix)
import numpy as np
covmatr = np.cov(X, rowvar=False)
print("Covariance Matrix by numpy=", covmatr)
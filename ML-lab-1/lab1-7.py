theta = ([2], [3], [3])
X = ([
    [1, 0, 2],
    [0, 1, 1],
    [2, 1, 0],
    [1, 1, 1],
    [0, 2, 1]
])
theta_product_X = []
for i in range(len(theta[0])):
    product = []
    for j in range(len(X)):
        s=0
        for k in range(len(X[0])):
            s=s+theta[k][i]*X[j][k]
        product.append(s)
    theta_product_X.append(product)
print(theta_product_X)



A=[[1,2,3],
   [4,5,6]]
A_transpose=[]
for i in range(len(A[0])):
    row=[]
    for j in range(len(A)):
        row.append(A[j][i])
    A_transpose.append(row)
print(A_transpose)
A_product_transpose=[]
for i in range(len(A[0])):
    product = []
    for j in range(len(A_transpose)):
        s=0
        for k in range(len(A)):
            s=s+A_transpose[i][k]*A[k][j]
        product.append(s)
    A_product_transpose.append(product)

print(A_product_transpose)







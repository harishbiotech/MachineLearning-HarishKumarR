#Compute the dot product of two vectors, x and y given below
#x = [2  1  2]T and y = [1  2  2]T . What is the meaning of the dot product of two vectors? Illustrate that with your own example.
x=[[2],[1],[2]]
y=[[1],[2],[2]]
dot_product=[]
for i in range(len(x[0])):
    add = 0
    for j in range(len(x)):
        add += x[j][i] * y[j][i]
    dot_product.append(add)
print("dot_product =", dot_product)
#here the dot product is 8
#The dot product of two vectors indicates how much one vector points in the direction of the other. If the dot product is positive
# the angle between them is acute; if it is zero, they are perpendicular; if it is negative, the angle between them is obtuse.
#here i get value is 8 so it is the angle between two vector is less then 90degree it is acute
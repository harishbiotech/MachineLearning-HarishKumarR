#===================Compute the derivative of a sigmoid function and visualize it=====================
import matplotlib.pyplot as plt
import math
x=[i for i in range(1,11)]
y=[]
def derivative_sigmoid(z):
    return (1/(1+math.exp(-z)))*(1-(1/(1+math.exp(-z))))
for values in x:
    y.append(derivative_sigmoid(values))

print(y)
plt.title("derivative sigmoid function")
plt.plot(x,y)
plt.show()
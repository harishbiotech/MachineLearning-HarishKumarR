#===========Implement sigmoid function in python and visualize it===============
import matplotlib.pyplot as plt
import math

x = [i for i in range(1, 101)]
y = []

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

for value in x:
    y.append(sigmoid(value))
print(x)
print(y)
plt.title("sigmoid function")
plt.plot(x,y)
plt.show()

import random as rnd
import matplotlib.pyplot as plt
x=[rnd.randint(-100,100) for x in range(100)]
print(x)
y=[2*i+3 for i in x]
print(y)
plt.plot(x,y)
plt.show()
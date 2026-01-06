import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-10,10,100)
y=2*np.square(x)+3*x+4
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("y=2x^2+3x+4")
plt.grid(True)
plt.show()
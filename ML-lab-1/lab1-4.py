#Gaussian PDF
#mean=0 sigma=15
import random as rnd
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-100, 100, 100)

m = 0      # mean
s = 15     # standard deviation

y = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-((x - m)**2) / (2 * s**2))

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Gaussian Probability Density Function")
plt.grid(True)
plt.show()



#Implement y = x1^2, plot x1, y in the range [start=--10, stop=10, num=100].
# Compute the value of derivatives at these points, x1 = -5, -3, 0, 3, 5.
# What is the value of x1 at which the function value (y) is zero. What do you infer from this?
import numpy as np
import matplotlib.pyplot as plt

# Define x range
x1 = np.linspace(-10, 10, 100)

# Define function y = x^2
y = x1**2

points = [-5, -3, 0, 3, 5]

for p in points:
    derivative = 2 * p
    print(f"dy/dx at x = {p} is {derivative}")
    
# Plot
plt.plot(x1, y)
plt.xlabel("x")
plt.ylabel("y = x^2")
plt.title("y = x^2")
plt.grid(True)
plt.show()




# Define the function
def y(x1, x2, x3):
    return 2*x1 + 3*x2 + 3*x3 + 4

# Gradient of y (constant for this linear function)
def gradient_y():
    dy_dx1 = 2
    dy_dx2 = 3
    dy_dx3 = 3  
    return dy_dx1, dy_dx2, dy_dx3

# Points at which to evaluate the gradient
points = [
    (0, 0, 0),
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
]

# Compute and print gradient at each point
for x1, x2, x3 in points:
    Y=y(x1, x2, x3)
    grad = gradient_y()
    print(f"Y = {Y}")
    print(f"At point (x1={x1}, x2={x2}, x3={x3})  gradient = {grad}")
    print("-----------------------------------")

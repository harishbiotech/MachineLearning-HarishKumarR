import numpy as np
x1 = np.array([3,6])
x2 = np.array([10,10])

def Transform(X):
    x1 = X[:, 0]
    x2 = X[:, 1]

    return np.column_stack((
        x1 ** 2,
        np.sqrt(2) * x1 * x2,
        x2 ** 2
    ))

phi_x1 = Transform(x1.reshape(1,-1))[0]
phi_x2 = Transform(x2.reshape(1,-1))[0]

dot_product = np.dot(phi_x1, phi_x2)

print("Dot product in transformed space:", dot_product)

def kernel(a,b):
    return (a[0]**2 * b[0]**2 +
            2*a[0]*b[0]*a[1]*b[1] +
            a[1]**2 * b[1]**2)

k_val = kernel(x1, x2)

print("Kernel value:", k_val)
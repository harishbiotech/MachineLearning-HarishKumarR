def Transform(X):
    x1 = X[:, 0]
    x2 = X[:, 1]

    return np.column_stack((
        x1 ** 2,
        np.sqrt(2) * x1 * x2,
        x2 ** 2
    ))
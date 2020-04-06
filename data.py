import numpy as np
from scipy.stats import multinomial


def etas(x):
    eta_1 = np.exp(-2 * x) * np.square(np.cos(4 * np.pi * x))
    eta_2 = (1 - x) * (1 - eta_1)
    eta_3 = x * (1 - eta_1)
    return eta_1, eta_2, eta_3


def generate_data(n, seed=None, x=None):
    if seed is not None:
        np.random.seed(seed)
    if x is None:
        x = np.random.uniform(size=(n, 1))

    eta_1, eta_2, eta_3 = etas(x)
    class_probs = np.hstack((eta_1, eta_2, eta_3))
    y_cats = np.array([
        multinomial.rvs(1,
                        class_probs[i],
                        random_state=(seed if i == 0 else None))
        for i in range(x.shape[0])
    ])
    y = np.argmax(y_cats, axis=1)
    return x, y


def load_dataset(path):
    with open(path) as in_f:
        data = np.array([[float(x) for x in line.strip().split(',')]
                         for line in in_f])
        X, y = (data[:, :-1], data[:, -1])
        return X, y

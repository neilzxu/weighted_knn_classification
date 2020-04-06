import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def _init_lambda(x):
    return min(int(np.ceil(5 * np.power(x, 1 / 3))), x)


class WeightedKNN:
    def set_weights(self, wknn_weights):
        self.wknn_weights = wknn_weights

    def __init__(self, wknn_weights, wknn_rate_fn=_init_lambda, **kwargs):
        self.knn = KNeighborsClassifier(**kwargs)
        self.wknn_weights, self.wknn_rate_fn = (wknn_weights, wknn_rate_fn)

    def fit(self, X, y):
        n_neighbors = self.wknn_rate_fn(X.shape[0])
        self.knn.set_params(n_neighbors=n_neighbors)
        self.knn.fit(X, y)

    def predict_logits(self, X):
        probs = self.knn.predict_proba(X)
        logits = probs * self.wknn_weights
        return logits

    def predict(self, X):
        return np.argmax(self.predict_logits(X), axis=1)

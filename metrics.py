from data import etas

import numpy as np


def hinge_mean(x):
    if len(x) == 0:
        return 0
    else:
        return np.sum(x)


def true_conf_matrix(wknn, granularity=1000):
    class_ct = wknn.knn.classes_.shape[0]
    X = np.arange(0, 1, 1 / granularity).reshape(-1, 1)
    eta_vals = np.hstack(etas(X))
    y_hat = wknn.predict_logits(X)
    best_logit = np.max(y_hat, axis=1)
    prediction = np.argmax(y_hat, axis=1)

    conf_matrices = []
    for class_idx in range(class_ct):
        pred_idxs = np.nonzero(prediction == class_idx)
        not_pred_idxs = np.nonzero(prediction != class_idx)
        class_eta = eta_vals[:, class_idx].squeeze()

        fp = hinge_mean(1 - class_eta[pred_idxs]) / X.shape[0]
        tp = hinge_mean(class_eta[pred_idxs]) / X.shape[0]
        tn = hinge_mean(1 - class_eta[not_pred_idxs]) / X.shape[0]
        fn = hinge_mean(class_eta[not_pred_idxs]) / X.shape[0]
        conf_matrices.append((tn, fn, fp, tp))

    return conf_matrices


def true_f1_score(wknn, granularity=1000):
    conf_matrices = true_conf_matrix(wknn, granularity)

    def f1(conf_matrix):
        tn, fn, fp, tp = conf_matrix
        denominator = 2 * tp + 1 * fn + fp
        return 0 if denominator == 0 else 2 * tp / denominator

    return f1(np.sum(np.stack(conf_matrices, axis=0), axis=0))

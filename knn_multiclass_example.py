"""

Code for the example in Section 5 of the paper.

"""
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

x = np.arange(0, 1, 0.001)
a = 2.
b = 2.
eta1 = np.exp(-a * x) * np.power(np.cos(2 * np.pi * b * x), 2)
eta2 = (1 - eta1) * (1 - x)
eta3 = (1 - eta1) * x
etas = np.array([eta1, eta2, eta3]).T
p = np.mean([eta1, eta2, eta3], axis=1)

## Plot the regression function
knn_folder = 'results/knn/figs/'
if not os.path.exists(knn_folder):
    os.makedirs(knn_folder)
regression_file = knn_folder + 'regression_01.png'
plt.rcParams.update({'font.size': 11})
regression_size = (3.25, 3)
fig = plt.figure(figsize=regression_size, dpi=300)
ax = fig.add_subplot()
ax.plot(x, eta1, label=r'$\eta_1(x)$', linestyle='-')
ax.plot(x, eta2, label=r'$\eta_2(x)$', linestyle='--')
ax.plot(x, eta3, label=r'$\eta_3(x)$', linestyle='-.')
ax.legend(loc='upper right')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\eta_c(x)$')
fig.savefig(regression_file, bbox_inches='tight')
plt.show()

## Weighted regression functions
i = 4
q0 = (0.8, 0.1, 0.1)
q1 = (0.1, 0.8, 0.1)
q2 = (0.1, 0.1, 0.8)
q3 = (0.2, 0.3, 0.5)
q4 = (0.5, 0.3, 0.2)
Q = [q0, q1, q2, q3, q4]
leg_locs = ['upper right'] * 3 + ['upper left'] + ['upper right']
weighted_file = knn_folder + 'weighted_0{0}.png'
plt.rcParams.update({'font.size': 11})
regression_size = (3.25, 3)
fig = plt.figure(figsize=regression_size, dpi=300)
ax = fig.add_subplot()
ax.plot(x, eta1 * Q[i][0], label=r'$q_1 \eta_1(x)$', linestyle='-')
ax.plot(x, eta2 * Q[i][1], label=r'$q_2 \eta_2(x)$', linestyle='--')
ax.plot(x, eta3 * Q[i][2], label=r'$q_3 \eta_3(x)$', linestyle='-.')
ax.legend(loc=leg_locs[i])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$q_c \eta_c(x)$')
fig.savefig(weighted_file.format(i), bbox_inches='tight')
plt.show()

# Plots demonstrating bound
j = 1
tqcx_file = knn_folder + 'tqcx0{0}.png'
q = (0.5, 0.3, 0.2)
q_p = (0.52, 0.29, 0.19)
q_pp = (0.48, 0.31, 0.21)
tqcx = (1 / q[0]) * np.max(np.array([eta2 * q[1], eta3 * q[2]]), axis=0)
tqcx_p = (1 / q_p[0]) * np.max(np.array([eta2 * q_p[1], eta3 * q_p[2]]),
                               axis=0)
tqcx_pp = (1 / q_pp[0]) * np.max(np.array([eta2 * q_pp[1], eta3 * q_pp[2]]),
                                 axis=0)

epsilons = [0.1, 0.05, 0.01, 0.001]
rqc = 2.
rqc_p = 2.
rqc_pp = 2.

lower_envelope = np.clip(tqcx_p - rqc_p * epsilons[j], 0, 1)
upper_envelope = np.clip(tqcx_pp + rqc_pp * epsilons[j], 0, 1)
in_between = np.logical_and(lower_envelope < eta1, eta1 < upper_envelope)

leg_locs = ['upper right'] * 3 + ['upper left'] + ['upper right']
weighted_file = knn_folder + 'figs/weighted_0{0}.png'
plt.rcParams.update({'font.size': 11})
regression_size = (3.25, 3)
fig = plt.figure(figsize=regression_size, dpi=300)
ax = fig.add_subplot()
ax.plot(x, eta1, label=r'$\eta_1(x)$', linestyle='-')
ax.plot(x,
        lower_envelope,
        label=r'$t^{\prime}(1, x) - \epsilon r^{\prime}(1)$',
        linestyle='--')
ax.plot(x,
        upper_envelope,
        label=r'$t^{\prime\prime}(1, x) + \epsilon r^{\prime \prime}(1)$',
        linestyle='-.')
ax.fill_between(x,
                0.,
                upper_envelope,
                where=in_between,
                alpha=0.5,
                color='red')
ax.legend(loc='upper right')
ax.set_xlabel(r'$x$')
fig.savefig(tqcx_file.format(j), bbox_inches='tight')
tne = np.mean((1 - eta1) * in_between)
tpe = np.mean(eta1 * in_between)
print("The estimated true negative error is {0}.".format(tne))
print("The estimated true positive error is {0}.".format(tpe))


## Computing the error
def joint_xy(n, alpha=2., beta=2.):
    x = np.random.uniform(size=n)
    eta1 = np.exp(-alpha * x) * np.power(np.cos(2 * np.pi * beta * x), 2)
    eta2 = (1 - eta1) * (1 - x)
    eta3 = (1 - eta1) * x
    etas = np.array([eta1, eta2, eta3])
    y = []
    for i in range(n):
        y.append(np.random.multinomial(1, etas[:, i]))

    return (x, np.array(y))


def pop_confusion_matrix(etas, q, c):

    etas_q = etas * q
    mqcx = np.max(np.delete(etas_q, c, axis=1), axis=1)
    negatives = etas_q[:, c] < mqcx
    positives = etas_q[:, c] >= mqcx

    truenegatives = np.mean((1 - etas[:, c]) * negatives)
    falsenegatives = np.mean(etas[:, c] * negatives)
    truepositives = np.mean(etas[:, c] * positives)
    falsepositives = np.mean((1 - etas[:, c]) * positives)

    return np.array([[truenegatives, falsepositives],
                     [falsenegatives, truepositives]])


q = (0.5, 0.3, 0.2)
n = 1000
m = 50
trials = 1000
x = np.arange(0, 1, 1. / n)
alpha = 2.
beta = 2.
eta1 = np.exp(-alpha * x) * np.power(np.cos(2 * np.pi * beta * x), 2)
eta2 = (1 - eta1) * (1 - x)
eta3 = (1 - eta1) * x
etas = np.array([eta1, eta2, eta3]).T
tqcx = (1 / q[0]) * np.max(np.array([eta2 * q[1], eta3 * q[2]]), axis=0)
c_of_interest = 0
pcm = pop_confusion_matrix(etas, q, c_of_interest)
abs_error_list = []
mcm_q_hat_list = []
for trial in range(trials):
    a, b = joint_xy(m)
    c = np.argmax(b, axis=1)
    k = int(5 * np.power(m, 1 / 3))
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(a.reshape(-1, 1), b)
    b_hat = knn.predict(a.reshape(-1, 1))
    c_pred = np.argmax(b_hat, axis=1)

    mcm = multilabel_confusion_matrix(c, c_pred) / m
    cq_pred = np.argmax(b_hat * q, axis=1)
    mcm_q_hat = multilabel_confusion_matrix(c, cq_pred) / m

    eta_q = etas * q
    eta_q_pred = np.argmax(eta_q, axis=1)
    mcm_pop = multilabel_confusion_matrix
    yhat = knn.predict(x.reshape(-1, 1))
    yhat_q = yhat * q
    yhat_q_pred = np.argmax(yhat_q, axis=1)
    mcm_q_tilde = multilabel_confusion_matrix(eta_q_pred,
                                              yhat_q_pred) / (3 * n)
    abs_errors = np.abs(pcm - mcm_q_hat[0])
    abs_error_list.append(abs_errors)
    mcm_q_hat_list.append(mcm_q_hat[0])
avg_error = np.mean(np.stack(abs_error_list), axis=0)
print(avg_error)

emp_error_file = knn_folder + "emp_error_{0}_{1}_{2}.png"
greater_than = np.sign(tqcx - eta_q[:, 0]) > np.sign(tqcx - yhat_q[:, 0])
less_than = np.sign(tqcx - eta_q[:, 0]) < np.sign(tqcx - yhat_q[:, 0])

fig = plt.figure(figsize=regression_size, dpi=300)
ax = fig.add_subplot()
ax.plot(x, eta_q[:, 0], label=r'$q_{1}\eta_1(x)$', linestyle='-')
ax.plot(x, yhat_q[:, 0], label=r'$q_{1} \hat{\eta}_{1}(x)$', linestyle='--')
ax.plot(x, tqcx, label=r'$t(q, 1, x)$', linestyle='-.')
ax.fill_between(x,
                0.,
                yhat_q[:, 0],
                where=greater_than,
                alpha=0.5,
                color='red')
ax.fill_between(x, 0., eta_q[:, 0], where=less_than, alpha=0.5, color='purple')
ax.legend(loc='upper right')
ax.set_xlabel(r'$x$')
fig.savefig(emp_error_file.format(n, m, k), bbox_inches='tight')
plt.show()

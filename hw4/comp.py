import matplotlib.pyplot as plt
import numpy as np


m = 10
with open('data.txt', 'r') as fp:
    y = fp.readline()
    y = np.array([float(substr) for substr in y.split()])
    s = fp.readline()
    s = np.array([float(substr) for substr in s.split()])

s_tilde = np.diff(s, prepend=0.)
z = np.reshape(y, newshape=(-1, m))
z = np.sum(z, axis=1)


def ols_estimator(X, Y):
    """
    X has dimensions (num observations, num input dimensions)
    Y has dimensions (num observations, num output dimensions)

    params has dimensions (num input dimensions, num output dimensions)
    """

    params = np.linalg.inv(X.T @ X) @ X.T @ Y
    errors = Y - X @ params

    return params, errors


# a(ii)
Chat, errors = ols_estimator(
    X=np.expand_dims(s_tilde, -1),
    Y=np.expand_dims(z, -1))

Rhat = np.var(errors) / m

print('Chat: ', Chat)
print('Rhat: ', Rhat)


def inv(M):
    return np.linalg.inv(M)


# b(ii)
from collections import defaultdict


def compute_potentials(y, A, C, Q, Rs, mu_0, Lambda_0):
    # compute potentials
    hs = dict()
    Js = defaultdict(dict)
    for i in range(len(y)):

        J_ii = C.T @ inv(Rs[i, :, :]) @ C
        if i == 0:
            J_ii += inv(Lambda_0)
        if i != 0:
            J_ii += inv(Q)
        if i != (len(y) - 1):
            J_ii += A.T @ inv(Q) @ A
        Js[i][i] = np.copy(J_ii)

        h_i = C.T @ inv(Rs[i]) @ y[i]
        if i == 0:
            h_i += inv(Lambda_0) @ mu_0
        hs[i] = np.copy(h_i)

        J_ij = - A.T @ inv(Q)
        if i == (len(y) - 1):
            J_ij *= 1
        Js[i][i + 1] = np.copy(J_ij)

    return hs, Js


def forward_pass(hs, Js):
    hs_forward = defaultdict(dict)
    Js_forward = defaultdict(dict)
    for i in range(len(hs) - 1):
        if i == 0:
            hs_forward[i][i + 1] = - Js[i][i + 1].T @ inv(Js[i][i]) @ hs[i]
            Js_forward[i][i + 1] = - Js[i][i + 1].T @ inv(Js[i][i]) @ Js[i][i + 1]
        else:
            hs_forward[i][i + 1] = - Js[i][i + 1].T @ inv(Js[i][i] + Js_forward[i - 1][i]) @ (
                        hs[i] + hs_forward[i - 1][i])
            Js_forward[i][i + 1] = - Js[i][i + 1].T @ inv(Js[i][i] + Js_forward[i - 1][i]) @ Js[i][i + 1]

    return hs_forward, Js_forward


def backward_pass(hs, Js):
    hs_backward = defaultdict(dict)
    Js_backward = defaultdict(dict)
    for i in range(len(hs) - 2, -1, -1):
        if i == (len(hs) - 2):
            hs_backward[i + 1][i] = - Js[i][i + 1] @ inv(Js[i + 1][i + 1]) @ hs[i + 1]
            Js_backward[i + 1][i] = - Js[i][i + 1] @ inv(Js[i + 1][i + 1]) @ Js[i][i + 1].T
        else:
            hs_backward[i + 1][i] = - Js[i][i + 1] @ inv(Js[i + 1][i + 1] + Js_backward[i + 2][i + 1]) @ (
                        hs[i + 1] + hs_backward[i + 2][i + 1])
            Js_backward[i + 1][i] = - Js[i][i + 1] @ inv(Js[i + 1][i + 1] + Js_backward[i + 2][i + 1]) @ Js[i][i + 1].T

    return hs_backward, Js_backward


def compute_posteriors(hs, hs_forward, hs_backward, Js, Js_forward, Js_backward):

    hs_hat, Js_hat = dict(), dict()
    for i in range(len(hs)):
        if i == 0:
            hs_hat[i] = hs[i] + hs_backward[i+1][i]
            Js_hat[i] = Js[i][i][0, 0] + Js_backward[i+1][i]
        elif i == (len(hs) - 1):
            hs_hat[i] = hs[i] + hs_forward[i-1][i]
            Js_hat[i] = Js[i][i] + Js_forward[i-1][i]
        else:
            hs_hat[i] = hs[i] + hs_forward[i-1][i] + hs_backward[i+1][i]
            Js_hat[i] = Js[i][i] + Js_forward[i - 1][i] + Js_backward[i + 1][i]

    return hs_hat, Js_hat


def gaussian_belief_prop(y, A, C, Q, Rs, mu_0, Lambda_0):
    """

    :param n: dimension of
    :param A:
    :param C:
    :param Q:
    :param R:
    :return:
    """

    # correct
    hs, Js = compute_potentials(y=y, A=A, C=C, Q=Q, Rs=Rs, mu_0=mu_0, Lambda_0=Lambda_0)

    hs_forward, Js_forward = forward_pass(hs=hs, Js=Js)
    hs_backward, Js_backward = backward_pass(hs=hs, Js=Js)

    hs_hat, Js_hat = compute_posteriors(
        hs=hs, hs_forward=hs_forward, hs_backward=hs_backward,
        Js=Js, Js_forward=Js_forward, Js_backward=Js_backward)

    times = np.arange(len(y))
    vars = np.array([inv(Js_hat[time])[0, 0] for time in times])
    means = np.array([(inv(Js_hat[time]) @ hs_hat[time])[0] for time in times])

    p = plt.plot(times, means)
    plt.fill_between(times,
                     means - np.sqrt(vars),
                     means + np.sqrt(vars),
                     color=p[-1].get_color(),
                     alpha=0.3)


# b(iii)
A = np.array([[0.9999]])
C = np.copy(Chat)
Q = np.array([[1.]])
Rs = np.full(shape=(len(y), 1, 1), fill_value=Rhat)
mu_0 = np.array([100.])
Lambda_0 = np.array([[1.]])
gaussian_belief_prop(y=np.expand_dims(y, axis=-1),
                     A=A, C=C, Q=Q, Rs=Rs, mu_0=mu_0, Lambda_0=Lambda_0)
plt.savefig('Comp_b(iii).jpg')
plt.show()


# b(iv)
A = np.array([[0.9999, 0.], [0.9999, 1.]])
C = np.array([[Chat[0, 0], 0.], [0., 1.]])
epsilon = 1e-6
sigma = 1e6
Q = np.array([[1., 1.], [1., 1. + epsilon]])
Rs = np.stack([np.array([[Rhat, 0.], [0., 0.]]) for _ in range(len(y))])
observed_indices = np.array([True if (i + 1) % m == 0 else False for i in range(len(y))])
Rs[observed_indices, 1, 1] = epsilon
Rs[np.logical_not(observed_indices), 1, 1] = sigma

s_rep = np.array([s[i//m] for i in range(len(y))])
y_aug = np.stack([np.squeeze(y), s_rep]).transpose((1, 0))
mu_0 = np.array([100., 100.])
Lambda_0 = np.array([[1., 0.], [0., 1.]])
gaussian_belief_prop(y=y_aug, A=A, C=C, Q=Q, Rs=Rs, mu_0=mu_0, Lambda_0=Lambda_0)
plt.savefig('Comp_b(iv).jpg')
plt.show()


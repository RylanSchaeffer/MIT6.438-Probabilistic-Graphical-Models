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
def compute_potentials(n, y, A, C, Q, R, mu_0, Lambda_0):

    from collections import defaultdict

    # compute potentials
    hs = dict()
    Js = defaultdict(dict)
    for i in range(len(y)):

        J_ii = C.T @ inv(R[i]) @ C
        if i == 0:
            J_ii += inv(Lambda_0)
        if i != 0:
            J_ii += inv(Q)
        if i != (len(y) - 1):
            J_ii += A.T @ inv(Q) @ A
        Js[i][i] = J_ii

        h_i = C.T @ inv(R) @ y[i]
        if i == 0:
            h_i += inv(Lambda_0) @ mu_0
        hs[i] = h_i

        J_ij = - A.T @ inv(Q)
        if i == (len(y) - 1):
            J_ij *= 1
        Js[i][i+1] = J_ij

    return hs, Js


def gaussian_belief_prop(n, y, A, C, Q, R, mu_0, Lambda_0):
    """

    :param n: dimension of
    :param A:
    :param C:
    :param Q:
    :param R:
    :return:
    """

    hs, Js = compute_potentials(n, y, A, C, Q, R, mu_0, Lambda_0)


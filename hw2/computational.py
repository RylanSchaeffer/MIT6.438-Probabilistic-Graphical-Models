import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


returns = pd.read_csv('returns.csv', header=None)

ksample_means = returns.mean().values
sample_cov = returns.cov().values
sample_prec = np.linalg.inv(sample_cov)
fig, ax = plt.subplots(nrows=1, ncols=1)
fig.suptitle('Sample Precision')
cax = ax.matshow(sample_prec)
fig.colorbar(cax)
ax.legend()
plt.savefig('compa.jpg')

# (a)(ii) Observations
# All covariances are positive, with smallest value 3.63e-05
# Many pairwise precisions are near 0, but not exactly 0
# Suggests that the underlying undirected graphical model is not sparse


# (b) Graphical Lasso
# As alpha increases in size, the covariance elements shrink towards 0
# Sparsity of graph also shrinks
from sklearn.covariance import graphical_lasso


penalized_covs, penalized_precs = {}, {}
for alpha in [1e-5, 1e-4, 1e-3]:
    penalized_cov, penalized_prec = graphical_lasso(
        emp_cov=sample_cov,
        alpha=alpha,
        max_iter=1000)
    penalized_covs[alpha] = penalized_cov
    penalized_precs[alpha] = penalized_prec
    print(f'Number of Non-Zero Edges (alpha = {alpha}): {np.sum(penalized_prec != 0.)}', )

fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].set_xlabel('Pairwise Covariances')
axes[1].set_xlabel('Pairwise Precisions')
for alpha, penalized_cov in penalized_covs.items():
    axes[0].hist(penalized_cov.flatten(),
                 label=r'$\alpha = $' + str(alpha),
                 bins=50)
    axes[1].hist(penalized_precs[alpha].flatten(),
                 label=r'$\alpha = $' + str(alpha),
                 bins=50)
axes[0].legend()
axes[1].legend()
plt.savefig('compb.jpg')
plt.show()


# (c) ii
J = np.eye(len(sample_cov))
n_iters = 20000
diffs = np.zeros(n_iters)
for n_iter in range(n_iters):

    assert np.all(np.diag(J) > 0)

    rand_indices = np.random.choice(
        np.arange(len(sample_cov)),
        size=2,
        replace=False)

    A_mask = np.full(len(sample_cov), fill_value=False)
    A_mask[rand_indices] = True
    B_mask = np.logical_not(A_mask)

    Saa_inv = np.linalg.inv(sample_cov[np.ix_(A_mask, A_mask)])
    Jaa = J[np.ix_(A_mask, A_mask)]
    Jab = J[np.ix_(A_mask, B_mask)]
    Jba = J[np.ix_(B_mask, A_mask)]
    Jbb = J[np.ix_(B_mask, B_mask)]
    L = Jab @ np.linalg.inv(Jbb) @ Jba
    Saa_inv_plus_L = Saa_inv + L
    if Saa_inv_plus_L[0, 1] <= 0:
        block_replacement = Saa_inv_plus_L
    else:
        L11, L22 = L[0, 0], L[1, 1]
        Suu = sample_cov[np.ix_(A_mask, A_mask)][0, 0]
        Svv = sample_cov[np.ix_(A_mask, A_mask)][1, 1]
        weird_numerator = (1 + np.sqrt(1 + 4 * Suu * Svv * L[0, 1] * L[1, 0]))
        block_replacement = np.array([
            [L11 + (weird_numerator / (2 * Suu)), 0.],
            [0., L22 + (weird_numerator / (2 * Svv))]])

    diff = np.linalg.norm(block_replacement - Jaa)
    print(f'Iteration {n_iter}: {diff}')
    diffs[n_iter] = diff
    J[np.ix_(A_mask, A_mask)] = block_replacement

plt.plot(np.arange(n_iters), diffs)
plt.savefig('compcii_convergence.jpg')
plt.show()


plt.savefig('compcii_precision.jpg')
fig, ax = plt.subplots(nrows=1, ncols=1)
fig.suptitle('Sample Precision')
cax = ax.matshow(J)
fig.colorbar(cax)
ax.legend()
plt.savefig('compcii_precision.jpg')
plt.show()

print('Number of Non-Zero Edges: ', np.sum(J != 0.))

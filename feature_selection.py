#!/usr/bin/env python3

import numpy as np
from gp_regression import GPR


def main():
    pass


def forward_selection(D, threshold=1e-2):
    """
    Greedy forward selection algorithm on dataset D=(X,Y).
    """
    X, Y = D
    inputs = np.array(X)
    n, m = X.shape
    selected_features = set()
    while True:
        stat_significance = np.zeros(m)
        for i in range(m):
            if i not in selected_features:
                index = [j for j in range(m) if j in selected_features.union({i})]
                new_X = X[:, index]
                stat_significance[i] = np.exp(compute_log_ml(GPR(initial_dataset=(new_X, Y))))
        anything_significant = np.any(stat_significant > threshold)
        if not anything_significant:
            break
        else:
            selected_features.add(np.argmax(stat_significance))
    return selected_features


def sq_exp_ard(M):
    """
    Squared exp kernel function for automatic relevance determination.
    """
    def f(x, y):
        x_ = x - y
        return np.exp(-0.5 * (x_.T @ M @ x_))
    return f


def sq_exp_ard_grad(gp):
    """
    Gradient with respect to each hyperparameter of the 
    squared exp kernel function for automatic relevance determination.
    """
    X, Y = gp.get_dataset()
    cov_mtx = gp.cov_mtx
    n, m = X.shape
    grad = np.tile(np.zeros((n, n)), (m, 1, 1))
    for i in range(n):
        for j in range(i + 1):
            x, y = X[i], X[j]
            x_sq = np.square(x - y)
            for k in range(m):
                grad[k, i, j] = -0.5 * x_sq[k] * cov_mtx[i, j]
                grad[k, j, i] = -0.5 * x_sq[k] * cov_mtx[j, i]
    return grad


def ard(D, threshold=1e-2, max_updates=100):
    """
    Automatic relevance determination algorithm on dataset D=(X,Y).
    """
    n = D[0].shape[1]
    l = np.ones(n)
    gp = GPR(cov_func=sq_exp_ard(np.diag(l)), initial_dataset=D, noise_variance=0)
    grad = compute_log_ml_grad(gp, sq_exp_ard_grad)
    num_updates = 0
    while np.sum(np.abs(grad)) > threshold or num_updates < max_updates:
        l -= learning_rate * grad
        gp = GPR(cov_func=sq_exp_ard(np.diag(l)), initial_dataset=D, noise_variance=0)
        grad = compute_log_ml_grad(gp, sq_exp_ard_grad)
        num_updates += 1
    return gp, l


def compute_log_ml(gp):
    """
    Computes the log of the marginal likelihood of the data given
    the hyperparameters of the model given by gp.
    """
    X, Y = gp.get_dataset()
    cov_mtx = gp._compute_noisy_cov()
    cov_inv = np.linalg.inv(cov_mtx)
    n = cov_mtx.shape[0]
    return 0.5 * (Y.T @ cov_inv @ Y - np.log(np.abs(cov_mtx)) - n * np.log(2 * np.pi))


def compute_log_ml_grad(gp, compute_cov_grad):
    """
    Computes the gradient of the log of the marginal likelihood
    with respect to the hyperparameters, using the function cov_grad
    to compute the gradient of the cov with respect to the hyperparameters.
    """
    X, Y = gp.get_dataset()
    cov_mtx = gp._compute_noisy_cov()
    cov_inv = np.linalg.inv(cov_mtx)
    cov_grad = compute_cov_grad(gp)
    alpha = cov_inv @ Y
    left_term = alpha @ alpha.T - cov_inv
    log_ml_grad = list()
    for i in range(len(cov_grad)):
        # TODO: only calculate terms in trace
        log_ml_grad.append(np.trace(left_term @ cov_grad[i]))
    return 0.5 * np.array(log_ml_grad)


if __name__ == "__main__":
    main()

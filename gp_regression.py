#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def main():
    plot_gpr_given_func(num_train=10)


def rbf(l_scale, sigma):
    c = -1.0 / (2.0 * np.power(l_scale, 2))
    def f(x, y):
        x_ = x - y
        return sigma * np.exp(c * (x_.T @ x_))
    return f


def plot_gpr_given_func(f=lambda x: np.sin(x), num_train=5):
    test_X = np.linspace(-5, 5, num=250) # 250 is arbitrary
    X = np.random.choice(test_X, num_train, replace=False).reshape((-1, 1))
    test_X = test_X.reshape((-1, 1))
    y = f(X)
    fig, ax = plt.subplots(1)
    ax.plot(X, y, 'b+')
    mean, variance, log_ml = gpr(X, y, rbf(1, 1), 1e-2, test_X)
    test_X = test_X.flatten()
    ax.plot(test_X, mean, 'red')
    ax.plot(test_X, f(test_X), 'green')
    plot_confidence(test_X, mean, np.sqrt(variance), ax)
    plt.show()


def plot_confidence(X, mean, std_dev, ax):
    ax.fill_between(X, mean - 2 * std_dev, mean + 2 * std_dev, facecolor='pink', alpha=0.4)


def gpr(inputs, targets, covariance_function, noise_level, test_inputs):
    """
    Implementation of Gaussian Process Regression (Algorithm 2.1 from Rasmussen).
    Expects all vector inputs as numpy arrays.
    """
    X, y, k, sigma_n, x = inputs, targets, covariance_function, noise_level, test_inputs
    K = compute_covariance(X, k)
    n = K.shape[0]
    L = np.linalg.cholesky(K + sigma_n * np.eye(n))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    m = test_inputs.size
    mean, variance = np.zeros(m), np.zeros(m)
    for i in range(m):
        test_input = test_inputs[i]
        test_cov = np.array([k(test_input, inputs[j]) for j in range(n)])
        v = np.linalg.solve(L, test_cov)
        mean[i] = test_cov.T @ alpha
        variance[i] = k(test_input, test_input) - v.T @ v
    log_marginal_likelihood = log_ml_helper(n, y, L, alpha)
    return mean, variance, log_marginal_likelihood


def log_ml_helper(n, y, L, alpha):
    return -0.5 * y.T @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)


def compute_covariance(X, k):
    """
    Compute covariance matrix for inputs X.
    Assumes covariance function k is symmetric.
    """
    n = X.shape[0]
    covariance = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            cov = k(X[i:i+1], X[j:j+1])
            covariance[i, j] = cov
            covariance[j, i] = cov
    return covariance


if __name__ == "__main__":
    main()

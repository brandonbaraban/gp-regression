#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.datasets import fetch_openml


def main():
    plot_gpr_given_func()
    plot_gpr_mauna_loa()


def load_mauna_loa_atmospheric_co2():
    """
    Taken from https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html.
    """
    ml_data = fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs


def plot_gpr_mauna_loa():
    """
    Some code borrowed from https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html.
    """
    X, y = load_mauna_loa_atmospheric_co2()
    X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
    noise_level = 5
    predict_and_plot(X, np.copy(y), rbf(l_scale=5), noise_level, X_)


def rbf(l_scale):
    def f(X, Y=None):
        if Y is None: # X with itself
            sqeuclidean = pdist(X / (l_scale ** 2), metric='sqeuclidean')
            K = np.exp(-0.5 * sqeuclidean)
            K = squareform(K) # convert from condensed to square
            np.fill_diagonal(K, 1)
        else: # X with Y
            sqeuclidean = cdist(X / (l_scale ** 2), Y / (l_scale ** 2), metric='sqeuclidean')
            K = np.exp(-0.5 * sqeuclidean)
        return K
    return f


def plot_gpr_given_func(f=lambda x: np.sin(x), num_train=5):
    test_X = np.linspace(-5, 5, num=250) # 250 is arbitrary
    X = np.random.choice(test_X, num_train, replace=False).reshape((-1, 1))
    y = f(X)
    noise_level = 0.0
    predict_and_plot(X, y, rbf(l_scale=1), noise_level, test_X, f)


def predict_and_plot(X, y, k, sigma_n, Xt, f=None):
    fig, ax = plt.subplots(1)
    if f is None:
        ax.plot(X, y, 'green') # plot f
    else:
        ax.plot(X, y, 'b+')
        ax.plot(Xt, f(Xt), 'green') # plot f
    mean, variance, log_ml = gpr(X, y, k, sigma_n, Xt.reshape((-1, 1)), normalize_y=True)
    print('log_ml', log_ml)
    ax.plot(Xt, mean, 'red') # plot MAP estimate of f
    v_noise = 1e-12 # add to variance just to account for floating point error
    plot_confidence(Xt.flatten(), mean, np.sqrt(variance + v_noise), ax)
    plt.show()


def plot_confidence(X, mean, std_dev, ax):
    ax.fill_between(X, mean - 2 * std_dev, mean + 2 * std_dev, facecolor='pink', alpha=0.4)


def gpr(X, y, k, sigma_n, Xt, normalize_y=False):
    """
    Implementation of Gaussian Process Regression (Algorithm 2.1 from Rasmussen).
    Expects all vector inputs as numpy arrays.
    """
    y_mean = 0.0
    if normalize_y: # make y zero-mean
        y_mean += np.mean(y, axis=0)
        y -= y_mean
    K = k(X)
    L = cholesky(K + sigma_n * np.eye(K.shape[0]), lower=True)
    alpha = cho_solve((L, True), y)
    test_cov = k(X, Xt)
    v = solve_triangular(L, test_cov, lower=True)
    mean = np.zeros(Xt.shape[0]) + np.reshape(test_cov.T @ alpha, (-1,)) + y_mean
    variance = np.ones(Xt.shape[0]) - np.sum(np.square(v), axis=0)
    log_marginal_likelihood = log_ml_helper(K.shape[0], y, L, alpha)
    return mean, variance, log_marginal_likelihood


def log_ml_helper(n, y, L, alpha):
    return np.asscalar(-0.5 * y.T @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi))


if __name__ == "__main__":
    main()

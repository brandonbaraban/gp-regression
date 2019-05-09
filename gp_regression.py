#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.datasets import fetch_openml


def main():
    plot_gpr_on_f(f=lambda x: np.sin(x), n_train=5, x_min=-np.pi, x_max=np.pi)
    # plot_gpr_mauna_loa()


# kernels


def rbf(l_scale):
    def f(X, Y=None):
        if Y is None: # X with itself
            sqeuclidean = pdist(X / l_scale, metric='sqeuclidean')
            K = np.exp(-0.5 * sqeuclidean)
            K = squareform(K) # convert from condensed to square
            np.fill_diagonal(K, 1)
        else: # X with Y
            sqeuclidean = cdist(X / l_scale, Y / l_scale, metric='sqeuclidean')
            K = np.exp(-0.5 * sqeuclidean)
        return K
    return f


# defaults
default_kernel = rbf(l_scale=1)
default_sigma_n = 1e-6
default_normalize_y = True


# plotting and testing functions


def plot_gpr_on_f(f, n_train, x_min, x_max):
    test_X = np.linspace(x_min, x_max, num=int((x_max - x_min) * 10)) # 10 is arbitrary
    X = np.random.choice(test_X, n_train, replace=False).reshape((-1, 1))
    y = f(X)
    noise_level = 0.0
    gpr_params = {'kernel': rbf(l_scale=1),
                  'sigma_n': 1e-6,
                  'normalize_y': -x_min != x_max}
    predict_and_plot(X, y, test_X, gpr_params, f)


def plot_gpr_mauna_loa():
    """
    Some code borrowed from https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html.
    """
    X, y = load_mauna_loa_atmospheric_co2()
    test_X = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
    gpr_params = {'kernel': rbf(l_scale=25),
                  'sigma_n': 5,
                  'normalize_y': True}
    predict_and_plot(X, y, test_X, gpr_params)


def predict_and_plot(X, y, Xt, gpr_params, f=None):
    fig, ax = plt.subplots(1)
    if f is None:
        ax.plot(X, y, 'green') # plot f
    else:
        ax.plot(X, y, 'b+')
        ax.plot(Xt, f(Xt), 'green') # plot f
    mean, variance, log_ml = gpr(X, y, Xt.reshape((-1, 1)), gpr_params)
    print('log_ml', log_ml)
    ax.plot(Xt, mean, 'red') # plot MAP estimate of f
    v_noise = 1e-12 # add to variance just to account for floating point error
    plot_confidence(Xt.flatten(), mean, np.sqrt(variance + v_noise), ax)
    plt.show()


def plot_confidence(X, mean, std_dev, ax):
    ax.fill_between(X, mean - 2 * std_dev, mean + 2 * std_dev, facecolor='pink', alpha=0.4)


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


# gpr functions


def extract_gpr_params(gpr_params):
    k = gpr_params.get('kernel', default_kernel)
    sigma_n = gpr_params.get('sigma_n', default_sigma_n)
    normalize_y = gpr_params.get('normalize_y', default_normalize_y)
    return k, sigma_n, normalize_y


def gpr(X, y, Xt, gpr_params):
    """
    Implementation of Gaussian Process Regression (Algorithm 2.1 from Rasmussen).
    Expects all vector inputs as numpy arrays.
    """
    k, sigma_n, normalize_y = extract_gpr_params(gpr_params)
    y_mean = 0.0
    if normalize_y: # make y zero-mean
        y_mean += np.mean(y, axis=0)
        y -= y_mean
    K = k(X)
    L = cholesky(K + sigma_n * np.eye(K.shape[0]), lower=True)
    alpha = cho_solve((L, True), y)
    test_cov = k(X, Xt)
    v = solve_triangular(L, test_cov, lower=True)
    mean = np.reshape(test_cov.T @ alpha, (-1,)) + y_mean
    variance = np.diag(k(Xt, Xt)) - np.sum(np.square(v), axis=0)
    log_marginal_likelihood = lml_helper(K.shape[0], y, L, alpha)
    y += y_mean
    return mean, variance, log_marginal_likelihood


def lml_helper(n, y, L, alpha):
    return np.asscalar(-0.5 * y.T @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi))


def lml_helper_cust(n, y, L, alpha):
    datafit = -0.5 * y.T @ alpha
    complexity = -np.sum(np.log(np.diag(L)))
    normalization = -0.5 * n * np.log(2 * np.pi)
    return np.asscalar(datafit), complexity, normalization


if __name__ == "__main__":
    main()

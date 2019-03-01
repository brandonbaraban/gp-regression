#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.datasets import fetch_openml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared


def main():
    plot_gpr_given_func()
    #plot_gpr_mauna_loa()


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

    k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
    k2 = 2.0**2 * RBF(length_scale=100.0) \
        * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                         periodicity_bounds="fixed")  # seasonal component
    k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
    k4 = 0.1**2 * RBF(length_scale=0.1) \
        + WhiteKernel(noise_level=0.1**2,
                      noise_level_bounds=(1e-3, np.inf))  # noise terms
    kernel = k1 + k2 + k3 + k4
    
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                                  normalize_y=True)
    gp.fit(X, y)
    
    print("\nLearned kernel: %s" % gp.kernel_)
    print("Log-marginal-likelihood: %.3f"
          % gp.log_marginal_likelihood(gp.kernel_.theta))

    X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
    y_pred, y_std = gp.predict(X_, return_std=True)
    
    plt.scatter(X, y, c='k')
    plt.plot(X_, y_pred)
    plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                     alpha=0.5, color='k')
    plt.xlim(X_.min(), X_.max())
    plt.xlabel("Year")
    plt.ylabel(r"CO$_2$ in ppm")
    plt.title(r"Atmospheric CO$_2$ concentration at Mauna Loa")
    plt.tight_layout()
    plt.show()


def rbf(l_scale, sigma):
    def f(X, Y=None):
        if Y is None: # X with itself
            sqeuclidean = pdist(X / (l_scale ** 2), metric='sqeuclidean')
            K = (sigma ** 2) * np.exp(-0.5 * sqeuclidean)
            K = squareform(K) # convert from condensed to square
            np.fill_diagonal(K, 1)
        else: # X with Y
            sqeuclidean = cdist(X / (l_scale ** 2), Y / (l_scale ** 2), metric='sqeuclidean')
            K = (sigma ** 2) * np.exp(-0.5 * sqeuclidean)
        return K
    return f


def plot_gpr_given_func(f=lambda x: np.sin(x), num_train=5):
    test_X = np.linspace(-5, 5, num=250) # 250 is arbitrary
    X = np.random.choice(test_X, num_train, replace=False).reshape((-1, 1))
    y = f(X)
    fig, ax = plt.subplots(1)
    ax.plot(X, y, 'b+')
    noise_level = 0.0
    mean, variance, log_ml = gpr(X, y, rbf(l_scale=1, sigma=1), noise_level, test_X.reshape((-1, 1)))
    ax.plot(test_X, mean, 'red') # plot MAP estimate of f
    ax.plot(test_X, f(test_X), 'green') # plot f
    v_noise = 1e-12 # add to variance just to account for floating point error
    plot_confidence(test_X, mean, np.sqrt(variance + v_noise), ax)
    plt.show()


def plot_confidence(X, mean, std_dev, ax):
    ax.fill_between(X, mean - 2 * std_dev, mean + 2 * std_dev, facecolor='pink', alpha=0.4)


def gpr(inputs, targets, covariance_function, noise_level, test_inputs, normalize_y=False):
    """
    Implementation of Gaussian Process Regression (Algorithm 2.1 from Rasmussen).
    Expects all vector inputs as numpy arrays.
    """
    X, y, k, sigma_n, Xt = inputs, targets, covariance_function, noise_level, test_inputs
    y_mean = 0.0
    if normalize_y: # make y zero-mean
        y_mean += np.mean(y, axis=0)
        y -= y_mean
    K = k(X)
    L = cholesky(K + sigma_n * np.eye(K.shape[0]), lower=True)
    alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True)) 
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

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
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
    c = -1.0 / (2.0 * np.power(l_scale, 2))
    def f(x, y):
        x_ = x - y
        return np.power(sigma, 2) * np.exp(c * (x_.T @ x_))
    return f


def plot_gpr_given_func(f=lambda x: np.sin(x), num_train=5):
    test_X = np.linspace(-5, 5, num=250) # 250 is arbitrary
    X = np.random.choice(test_X, num_train, replace=False).reshape((1, -1))
    y = f(X)
    fig, ax = plt.subplots(1)
    ax.plot(X, y, 'b+')
    noise_level = 0.0
    mean, variance, log_ml = gpr(X, y.T, rbf(l_scale=1, sigma=1), noise_level, test_X.reshape((1, -1)))
    ax.plot(test_X, mean, 'red') # plot MAP estimate of f
    ax.plot(test_X, f(test_X), 'green') # plot f
    v_noise = 1e-12 # add to variance just to account for floating point error
    plot_confidence(test_X, mean, np.sqrt(variance + v_noise), ax)
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
    L = linalg.cholesky(K + sigma_n * np.eye(n), lower=True, check_finite=False)
    alpha = linalg.solve_triangular(L.T, 
                                    linalg.solve_triangular(L, y, lower=True, check_finite=False), 
                                    check_finite=False)
    m = test_inputs.size
    mean, variance = np.zeros(m), np.zeros(m)
    for i in range(m):
        test_input = test_inputs[:, i]
        test_cov = np.array([k(test_input, inputs[:, j]) for j in range(n)])
        v = linalg.solve_triangular(L, test_cov, lower=True, check_finite=False)
        mean[i] = test_cov.T @ alpha
        variance[i] = k(test_input, test_input) - v.T @ v
    log_marginal_likelihood = log_ml_helper(n, y, L, alpha)
    return mean, variance, log_marginal_likelihood


def log_ml_helper(n, y, L, alpha):
    return np.asscalar(-0.5 * y.T @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi))


def compute_covariance(X, k):
    """
    Compute covariance matrix for inputs X.
    Assumes covariance function k is symmetric.
    """
    n = X.shape[1]
    covariance = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            cov = k(X[:, i:i+1], X[:, j:j+1])
            covariance[i, j] = cov
            covariance[j, i] = cov
    return covariance


if __name__ == "__main__":
    main()

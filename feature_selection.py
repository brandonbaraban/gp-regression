#!/usr/bin/env python3

import time
import numpy as np
import matplotlib.pyplot as plt
from gp_regression import gpr, rbf, lml_helper, lml_helper_cust
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_boston


# kernels
default_rbf = rbf(l_scale=1)


# functions
simple_sin = lambda x: np.sin(x[:, 0:1])
less_simple_sin = lambda x: 9 * np.sin(x[:, 0:1]) + \
                            5 * np.sin(x[:, 1:2]) +  \
                            2 * np.sin(x[:, 2:3])


# defaults
default_eta = 1e-2
default_sigma_n = 1e-6
default_init_min = 20
default_init_max = 35
default_max_iters = 100


def main():
    pass
    # print('timing as function of n...')
    # time_gfs_ard_n(f=simple_sin,
    #                  n_features=2,
    #                  init_min=-np.pi,
    #                  init_max=np.pi,
    #                  params={'eta': 1e-6,
    #                          'sigma_n': 1e-6,
    #                          'normalize_y': False,
    #                          'init_min': 25, 
    #                          'init_max': 25,
    #                          'max_iters': 10},
    #                  n_max=100)
    # print('timing as function of d...')
    # time_gfs_ard_d(f=simple_sin,
    #                  n_train=10,
    #                  init_min=-np.pi,
    #                  init_max=np.pi,
    #                  params={'eta': 1e-6,
    #                          'sigma_n': 1e-6,
    #                          'normalize_y': False,
    #                          'init_min': 25, 
    #                          'init_max': 25,
    #                          'max_iters': 10},
    #                  n_max=100)
    # test_gfs_ard_on_f(f=less_simple_sin,
    #                  n_train=100, 
    #                  n_val=20,
    #                  n_test=100,
    #                  n_features=6,
    #                  init_min=-np.pi,
    #                  init_max=np.pi,
    #                  params={'eta': 1e-6,
    #                          'sigma_n': 1e-6,
    #                          'normalize_y': False,
    #                          'init_min': 25, 
    #                          'init_max': 25,
    #                          'max_iters': 20})
    # test_boston_house_prices(n_train=300,
    #                           params={'eta': 1e-8,
    #                                   'sigma_n': 1e-6,
    #                                   'normalize_y': False,
    #                                   'init_min': 25, 
    #                                   'init_max': 25,
    #                                   'max_iters': 1000})


# testing and plotting


def time_gfs_ard_n(f, n_features, init_min, init_max, params, n_max):
    all_X, all_y = train_data_generator(f, n_max, n_features, init_min, init_max)
    ard_times = list()
    gfs_times = list()
    k = rbf(l_scale=1)
    for n_train in range(1, n_max + 1):
        X, y = all_X[:n_train], all_y[:n_train]
        ard_times.append(time_ard(X, y, params))
        gfs_times.append(time_gfs(k, X, y, params.get('sigma_n', default_sigma_n)))
    f, ax1 = plt.subplots(1)
    ax1.plot(gfs_times, 'r', label='gfs')
    ax1.plot(ard_times, 'b', label='ard')
    ax1.set_xlabel('number of training points (n)')
    ax1.set_ylabel('runtime')
    ax1.legend()
    plt.tight_layout()
    plt.show()


def time_gfs_ard_d(f, n_train, init_min, init_max, params, n_max):
    all_X, all_y = train_data_generator(f, n_train, n_max, init_min, init_max)
    ard_times = list()
    gfs_times = list()
    k = rbf(l_scale=1)
    for n_features in range(1, n_max + 1):
        X, y = all_X[:, :n_train], all_y[:, :n_train]
        ard_times.append(time_ard(X, y, params))
        gfs_times.append(time_gfs(k, X, y, params.get('sigma_n', default_sigma_n)))
    f, ax1 = plt.subplots(1)
    ax1.plot(gfs_times, 'r', label='gfs')
    ax1.plot(ard_times, 'b', label='ard')
    ax1.set_xlabel('number of features (d)')
    ax1.set_ylabel('runtime')
    ax1.legend()
    plt.tight_layout()
    plt.show()



def time_ard(X, y, params):
    start = time.time()
    ard(X, y, params)
    end = time.time()
    return end - start


def time_gfs(k, X, y, sigma_n):
    start = time.time()
    gfs(k, X, y, sigma_n)
    end = time.time()
    return end - start


def test_gfs_ard_on_f(f, n_train, n_val, n_test, n_features, 
                     init_min, init_max, params):
    data_X, data_y = data_generator(f, n_train, n_val, n_test, n_features, init_min, init_max)
    gfs_vs_ard(data_X, data_y, params)


def test_boston_house_prices(n_train, params):
    X, y = shuffle_data(*load_boston(return_X_y=True))
    n = X.shape[0]
    n_val = (n - n_train) // 2
    train_X, train_y = X[:n_train, :], y[:n_train, :]
    val_X, val_y = X[n_train: n_train + n_val, :], y[n_train: n_train + n_val, :]
    test_X, test_y = X[n_train + n_val:, :], y[n_train + n_val:, :]
    gfs_vs_ard((train_X, val_X, test_X), (train_y, val_y, test_y), params)
    # gfs_vs_lard((train_X, val_X, test_X), (train_y, val_y, test_y), params)


def gfs_vs_ard(data_X, data_y, params):
    theta, ard_lml, datafit, complexity, thetas = test_ard(data_X, data_y, params)
    ard_features = np.array([i[0] for i in sorted(enumerate(theta), key=lambda x: x[1])])
    print('ard output', theta)
    print('ard sorted features', ard_features)
    gfs_features, gfs_lml = test_gfs(data_X[0], data_y[0], 
                                     params.get('sigma_n', default_sigma_n))
    print('gfs sorted features', gfs_features)
    gfs_mse, ard_mse = features_vs_mse(gfs_features, ard_features, data_X, data_y, params)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(gfs_mse, 'r+', label='gfs')
    ax1.plot(ard_mse, 'b', label='ard')
    ax1.set_xlabel('number of features used')
    ax1.set_ylabel('mean squared error')
    ax1.legend()
    colors = 'bgrcmyk'
    styles = ['-', '--', ':']
    for i in range(thetas.shape[0]):
        ax2.plot(thetas[i], colors[i % len(colors)] + styles[i // len(colors)], label=f'theta_{i}')
    ax2.set_xlabel('time')
    ax2.set_ylabel('theta_i')
    ax2.legend()
    ax3.plot(gfs_lml, 'r', label='gfs')
    ax3.set_xlabel('time')
    ax3.set_ylabel('log_ml')
    ax3.legend()
    ax4.plot(ard_lml, 'b', label='ard')
    ax4.set_xlabel('time')
    ax4.set_ylabel('log_ml')
    ax4.legend()
    # ax5.plot(datafit, 'r', label='datafit')
    # ax5.set_xlabel('time')
    # ax5.set_ylabel('datafit')
    # ax6.plot(complexity, 'b', label='complexity')
    # ax6.set_xlabel('time')
    # ax6.set_ylabel('complexity')
    plt.tight_layout()
    plt.show()


def gfs_vs_lard(data_X, data_y, params):
    """
    LARD = Lazy ARD.
    """
    sigma_n = params.get('sigma_n', default_sigma_n)
    init_min = params.get('init_min', default_init_min)
    init_max = params.get('init_max', default_init_max)
    lard_features = lazy_ard(data_X[0], data_y[0], sigma_n, init_min, init_max)
    print('lard sorted features', lard_features)
    gfs_features, gfs_lml = test_gfs(data_X[0], data_y[0], 
                                     params.get('sigma_n', default_sigma_n))
    print('gfs sorted features', gfs_features)
    gfs_mse, lard_mse = features_vs_mse(gfs_features, lard_features, data_X, data_y, params)
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(gfs_mse, 'r', label='gfs')
    ax1.plot(lard_mse, 'b', label='lard')
    ax1.set_xlabel('number of features used')
    ax1.set_ylabel('mean squared error')
    ax1.legend()
    ax2.plot(gfs_lml, 'r', label='gfs')
    ax2.set_xlabel('time')
    ax2.set_ylabel('log_ml')
    ax2.legend()
    plt.tight_layout()
    plt.show()


def plot_log_marginal_likelihood(X, y, sigma_n, theta_min, theta_max, delta=1):
    """
    Props to TS. Assumes X is two-dimensional dataset.
    """
    theta0s = np.arange(theta_min, theta_max, delta)
    theta1s = np.arange(theta_min, theta_max, delta)
    theta0, theta1 = np.meshgrid(theta0s, theta1s)
    f = lambda th0, th1 : log_marginal_likelihood(X, y, sigma_n, np.array([th0, th1]))
    v_func = np.vectorize(f)

    print("calculating log marginal-likelihoods...")
    lml = v_func(theta0, theta1)
    print("done")

    plt.figure()
    plt.contourf(theta0, theta1, lml, levels=20, cmap='RdGy')
    plt.colorbar()
    plt.title("log_ml")
    plt.xlabel("theta0")
    plt.ylabel("theta1")
    plt.show()


# greedy forward selection


def gfs(k, X, y, sigma_n):
    """
    Greedy forward selection algorithm on dataset (X, y).
    Uses marginal likelihood of data on kernel k.
    Expects vector inputs as numpy arrays.
    """
    n, d = X.shape
    selected_features = set()
    selected_features_ordered = list()
    prev_lml = 0
    while len(selected_features) < d:
        stat_sig = np.zeros(d) + np.NINF
        for i in range(d):
            if i not in selected_features:
                index = [j for j in range(d) if j in selected_features.union({i})]
                new_X = X[:, index]
                lml = get_lml(k, new_X, y, sigma_n)
                stat_sig[i] = lml - prev_lml
        max_index = np.argmax(stat_sig)
        prev_lml += stat_sig[max_index]
        selected_features.add(max_index)
        selected_features_ordered.append(max_index)
    return np.array(selected_features_ordered), np.array(lml)


def test_gfs(X, y, sigma_n, num_features=0):
    """
    Greedy forward selection algorithm on dataset (X, y).
    Expects vector inputs as numpy arrays.
    Includes plotting/debugging structures/statements.
    """
    n, d = X.shape
    if num_features == 0:
        num_features = d
    k = rbf(l_scale=1)
    selected_features = set()
    selected_features_ordered = list()
    lmls = list()
    prev_lml = 0
    while len(selected_features) < num_features:
        stat_sig = np.zeros(d) + np.NINF
        for i in range(d):
            if i not in selected_features:
                index = [j for j in range(d) if j in selected_features.union({i})]
                new_X = X[:, index]
                lml = get_lml(k, new_X, y, sigma_n)
                stat_sig[i] = lml - prev_lml
        max_index = np.argmax(stat_sig)
        prev_lml += stat_sig[max_index]
        lmls.append(prev_lml)
        selected_features.add(max_index)
        selected_features_ordered.append(max_index)
    return np.array(selected_features_ordered), np.array(lmls)


# ard functions


def ard(X, y, params):
    """
    Automatic relevance determination algorithm.
    Optimization using vanilla gradient descent.
    """
    eta, sigma_n, init_min, init_max, max_iters = extract_ard_params(params)
    n, d = X.shape
    theta = (init_max - init_min) * np.random.random(d) + init_min
    t = 0
    while t < max_iters:
        t += 1
        grad = rbf_ard_gradient(theta, X, y, sigma_n)
        theta += grad * eta
    return theta


def test_ard(data_X, data_y, params):
    """
    Automatic relevance determination algorithm.
    Optimization using vanilla gradient descent.
    Includes plotting/debugging structures/statements.
    """
    X, val_X, _ = data_X
    y, val_y, _ = data_y
    eta, sigma_n, init_min, init_max, max_iters = extract_ard_params(params)
    n, d = X.shape
    theta = (init_max - init_min) * np.random.random(d) + init_min
    init_theta = theta.copy()
    t = 0
    lmls = list()
    datafits = list()
    complexities = list()
    thetas = [init_theta.copy()]
    while t < max_iters:
        t += 1
        grad, (datafit, complexity, norm) = test_rbf_ard_gradient(theta, X, y, sigma_n)
        print('datafit', datafit, 'complexity', complexity, 'normalization', norm)
        # num_grad = rbf_ard_num_gradient(theta, X, y, sigma_n)
        # grad_incorrect = np.any(np.abs(grad - num_grad) > 1e-1)
        # print('grad incorrect?', grad_incorrect)
        # if grad_incorrect:
        #     print('my_grad', grad)
        #     print('nu_grad', num_grad)
        # else:
        print('grad', grad)
        print('theta', theta)
        step = grad.copy()
        for i in range(step.size):
            if np.abs(step[i]) > 1e-2:
                step[i] *= eta
        # theta += grad * eta
        theta += step
        lmls.append(datafit + complexity + norm)
        datafits.append(datafit)
        complexities.append(complexity)
        thetas.append(theta.copy())
        if t > 1:
            print('delta log_ml')
            print(lmls[-1] - lmls[-2])
            print()
        if np.abs(grad)[0] < 1e-2:
            break
    print('init_theta', init_theta)
    return theta, np.array(lmls), np.array(datafits), np.array(complexities), np.array(thetas).T


def rbf_ard_gradient(theta, X, y, sigma_n):
    """
    Gradient of RBF for ARD with respect to each hyper-parameter.
    """
    n, d = X.shape
    K = rbf_ard(theta)(X)
    L = cholesky(K + sigma_n * np.eye(K.shape[0]), lower=True)
    L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
    K_inv = L_inv.T @ L_inv
    alpha = K_inv @ y
    K_grad = np.tile(K, (d, 1, 1))
    dtheta = 1.0 / np.power(theta, 3)
    for k in range(d):
        for i in range(n):
            for j in range(i + 1):
                c = np.square(X[i, k] - X[j, k]) * dtheta[k]
                K_grad[k, i, j] *= c
                K_grad[k, j, i] *= c
    c = alpha @ alpha.T - K_inv
    return 0.5 * np.array([np.trace(c @ dK) for dK in K_grad])


def test_rbf_ard_gradient(theta, X, y, sigma_n):
    """
    Gradient of RBF for ARD with respect to each hyper-parameter.
    """
    n, d = X.shape
    K = rbf_ard(theta)(X)
    L = cholesky(K + sigma_n * np.eye(K.shape[0]), lower=True)
    L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
    K_inv = L_inv.T @ L_inv
    alpha = K_inv @ y
    K_grad = np.tile(K, (d, 1, 1))
    dtheta = 1.0 / np.power(theta, 3)
    for k in range(d):
        for i in range(n):
            for j in range(i + 1):
                c = np.square(X[i, k] - X[j, k]) * dtheta[k]
                K_grad[k, i, j] *= c
                K_grad[k, j, i] *= c
    c = alpha @ alpha.T - K_inv
    return 0.5 * np.array([np.trace(c @ dK) for dK in K_grad]), lml_helper_cust(n, y, L, alpha)


def rbf_ard_num_gradient(theta, X, y, sigma_n, delta=1e-6):
    g = np.zeros(theta.shape)
    for i in range(theta.shape[0]):
        thi = theta[i]
        theta[i] = thi - delta
        lml_thm = get_lml_ard(theta, X, y, sigma_n)
        theta[i] = thi + delta
        lml_thp = get_lml_ard(theta, X, y, sigma_n)
        theta[i] = thi
        g[i] = (lml_thp - lml_thm) / (2 * delta)
    return g


def lazy_ard(X, y, sigma_n, init_min, init_max):
    """
    For fun.
    """
    n, d = X.shape
    theta = (init_max - init_min) * np.random.random(d) + init_min
    grad, lml = rbf_ard_gradient(theta, X, y, sigma_n)
    return np.array([i[0] for i in sorted(enumerate(grad), key=lambda x: x[1])])


def get_lml_ard(theta, X, y, sigma_n):
    """
    Get log marginal likelihood of data on ARD kernel given theta.
    """
    K = rbf_ard(theta)(X)
    L = cholesky(K + sigma_n * np.eye(K.shape[0]), lower=True)
    alpha = cho_solve((L, True), y)
    return lml_helper(K.shape[0], y, L, alpha)


def rbf_ard(theta):
    """
    RBF for automatic relevance determination.
    """
    def f(X):
        sqeuclidean = pdist(X / theta, metric='sqeuclidean')
        K = np.exp(-0.5 * sqeuclidean)
        K = squareform(K) # convert from condensed to square
        np.fill_diagonal(K, 1)
        return K
    return f


def extract_ard_params(params):
    eta = params.get('eta', default_eta)
    sigma_n = params.get('sigma_n', default_sigma_n)
    init_min = params.get('init_min', default_init_min)
    init_max = params.get('init_max', default_init_max)
    max_iters = params.get('max_iters', default_max_iters)
    return eta, sigma_n, init_min, init_max, max_iters


# helpers


def get_lml(k, X, y, sigma_n):
    """
    Get log marginal likelihood of data on kernel k.
    """
    K = k(X)
    L = cholesky(K + sigma_n * np.eye(K.shape[0]), lower=True)
    alpha = cho_solve((L, True), y)
    return lml_helper(K.shape[0], y, L, alpha)


def data_generator(f, n_train, n_val, n_test, n_dim, x_min, x_max):
    n = n_train + n_val + n_test
    X = (x_max - x_min) * np.random.random((n, n_dim)) + x_min
    train_X, val_X, test_X = X[:n_train, :], X[n_train: n_val, :], X[n_val:, :] 
    return (train_X, val_X, test_X), (f(train_X), f(val_X),f(test_X))


def train_data_generator(f, n_train, n_dim, x_min, x_max):
    X = (x_max - x_min) * np.random.random((n_train, n_dim)) + x_min
    return X, f(X)


def shuffle_data(X, y):
    data = np.hstack((X, y.reshape((-1, 1))))
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1:]


def features_vs_mse(gfs_features, ard_features, data_X, data_y, gpr_params):
    d = ard_features.size
    gfs_mse = np.zeros(d)
    ard_mse = np.zeros(d)
    train_X, _, test_X = data_X
    train_y, _, test_y = data_y
    for i in range(d):
        gfs_i = gfs_features[:i + 1]
        ard_i = ard_features[:i + 1]
        gfs_mse[i] = gpr_mse(train_X[:, gfs_i], train_y, 
                                test_X[:, gfs_i], test_y, gpr_params)
        ard_mse[i] = gpr_mse(train_X[:, ard_i], train_y, 
                                 test_X[:, ard_i], test_y, gpr_params)
    return gfs_mse, ard_mse


def gpr_mse(train_X, train_y, test_X, test_y, gpr_params):
    mean, var, lml = gpr(train_X, train_y, test_X, gpr_params)
    mse = np.mean(np.square(mean.reshape((-1, 1)) - test_y), axis=0)
    return mse


if __name__ == "__main__":
    main()

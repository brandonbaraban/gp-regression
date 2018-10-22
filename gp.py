#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def main():
    #plot_sample_from_gp()
    plot_sample_from_gp_given_func(num_train=5, num_samples=5)


def plot_sample_from_gp(num_samples=1):
    X = np.linspace(-5, 5, num=250)
    gp = GP(cov_func=SQUARED_EXP)
    for i in range(num_samples):
        mean, covariance = gp.predict(X)
        Y = np.random.multivariate_normal(mean, covariance)
        plt.plot(X, Y)
    plt.show()


def plot_sample_from_gp_given_func(f=lambda x: x, num_train=5, num_samples=1):
    X = np.linspace(-5, 5, num=250)
    train_X = np.random.choice(X, num_train, replace=False)
    train_Y = [f(x) for x in train_X]
    gp = GP(cov_func=SQUARED_EXP, initial_dataset=(train_X, train_Y))
    for i in range(num_samples):
        mean, covariance = gp.predict(X)
        Y = np.random.multivariate_normal(mean, covariance)
        plt.plot(X, Y, 'r')
    plt.plot(train_X, train_Y, 'b+')
    plt.show()


DEFAULT_MEAN = lambda x: 0
DEFAULT_COVARIANCE = lambda x, y: int(x == y)
SQUARED_EXP = lambda x, y: np.exp(-0.5 * pow(np.linalg.norm(x-y), 2))
class GP:
    """ A representation of a Gaussian Process.  """

    def __init__(self, 
                mean_func=DEFAULT_MEAN, 
                cov_func=DEFAULT_COVARIANCE,
                initial_dataset=None):
        self.mean_func = mean_func
        self.cov_func = cov_func
        self.cov_mtx = None
        self.dataset = (list(), list())
        if initial_dataset is not None:
            X, Y = initial_dataset
            self.update(X, Y)

    def update(self, new_X, new_Y):
        X, Y = self.dataset
        X.extend(new_X)
        Y.extend(new_Y)
        self.cov_mtx = compute_cov_mtx(X, self.cov_func)

    def predict(self, X):
        """ 
        Returns the mean and variance of the 
        multivariate Guassian distribution describing f(X).
        """
        old_X, old_Y = self.dataset
        cov_mtx = compute_cov_mtx(X, self.cov_func)
        if len(old_X) == 0:
            return [self.mean_func(x) for x in X], cov_mtx
        cov_rows = compute_cov_rows(X, old_X, self.cov_func)
        cov_cols = compute_cov_cols(X, old_X, self.cov_func)
        cov_inv = np.linalg.inv(self.cov_mtx)
        return cov_rows @ cov_inv @ np.array(old_Y).T, \
                cov_mtx - cov_rows @ cov_inv @ cov_cols


def compute_cov(new_X, X, old_cov, cov_func):
    cov_mtx = compute_cov_mtx(new_X, cov_func)
    if old_cov is None:
        return cov_mtx
    new_rows = compute_cov_rows(new_X, X, cov_func)
    new_cols = compute_cov_cols(new_X, X, cov_func)
    return np.hstack((np.vstack((old_cov, new_rows)), np.vstack((new_cols, cov_mtx))))


def compute_cov_rows(new_X, old_X, cov_func):
    new_rows = []
    for new_x in new_X:
        new_row = []
        for old_x in old_X:
            new_row.append(cov_func(old_x, new_x))
        new_rows.append(new_row)
    return np.array(new_rows)


def compute_cov_cols(new_X, old_X, cov_func):
    new_cols = []
    for new_x in new_X:
        new_col = []
        for old_x in old_X:
            new_col.append(cov_func(new_x, old_x))
        new_cols.append(new_col)
    return np.array(new_cols).T


def compute_cov_mtx(X, cov_func):
    cov_mtx = []
    for x in X:
        row = []
        for y in X:
            row.append(cov_func(x, y))
        cov_mtx.append(row)
    return np.array(cov_mtx)


if __name__ == "__main__":
    main()

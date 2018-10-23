#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def main():
    #plot_sample_from_gp()
    plot_sample_from_gp_given_func(f=lambda x: np.sin(x), num_train=10, num_samples=3)


def plot_sample_from_gp(num_samples=1):
    X = np.linspace(-5, 5, num=250)
    gp = GPR(cov_func=SQUARED_EXP)
    fig, ax = plt.subplots(1)
    mean, std_dev, covariance = gp.predict(X)
    for i in range(num_samples):
        Y = gp.sample(X)
        ax.plot(X, Y, 'red')
    plot_confidence(X, mean, std_dev, ax)
    plt.show()


def plot_sample_from_gp_given_func(f=lambda x: x, num_train=5, num_samples=1):
    X = np.linspace(-5, 5, num=250)
    train_X = np.random.choice(X, num_train, replace=False)
    train_Y = [f(x) for x in train_X]
    gp = GPR(cov_func=SQUARED_EXP, initial_dataset=(train_X, train_Y))
    fig, ax = plt.subplots(1)
    mean, std_dev, covariance = gp.predict(X)
    ax.plot(train_X, train_Y, 'b+')
    ax.plot(X, mean, 'green')
    for i in range(num_samples):
        Y = gp.sample(X)
        #ax.plot(X, Y, 'red')
    plot_confidence(X, mean, std_dev, ax)
    plt.show()


def plot_confidence(X, mean, std_dev, ax):
    ax.fill_between(X, mean - 2 * std_dev, mean + 2 * std_dev, facecolor='pink', alpha=0.4)


ZERO_MEAN = lambda x: 0
ZERO_COVARIANCE = lambda x, y: int(x == y)
SQUARED_EXP = lambda x, y: np.exp(-0.5 * pow(np.linalg.norm(x-y), 2))
class GPR(object):
    """ A representation of a Gaussian Process.  """

    def __init__(self, 
                mean_func=ZERO_MEAN, 
                cov_func=SQUARED_EXP,
                initial_dataset=None,
                noisy=True):
        self.mean_func = mean_func
        self.cov_func = cov_func
        self.noise_variance = 1e-2 * int(noisy)
        self.cov_mtx = None
        self.dataset = (list(), list())
        if initial_dataset is not None:
            X, Y = initial_dataset
            self.update(X, Y)

    def update(self, new_X, new_Y):
        """
        Updates the GP to reflect the posterior given new training points.
        """
        X, Y = self.dataset
        self._compute_cov(new_X)
        X.extend(new_X)
        Y.extend(new_Y)

    def get_distribution(self, new_X):
        """ 
        Returns the mean and covariance of the 
        multivariate Guassian distribution describing f(new_X).
        """
        X, Y = self.dataset
        cov_mtx = self._compute_cov_mtx(new_X)
        if self.cov_mtx is None:
            n = cov_mtx.shape[0]
            return np.array([self.mean_func(x) for x in new_X]), cov_mtx + self.noise_variance * np.eye(n)
        n = self.cov_mtx.shape[0]
        cov_ext = self._compute_cov_ext(new_X)
        cov_inv = np.linalg.inv(self.cov_mtx + self.noise_variance * np.eye(n))
        return np.array(cov_ext @ cov_inv @ np.array(Y).T), \
                np.array(cov_mtx - cov_ext @ cov_inv @ cov_ext.T)

    def predict(self, X):
        """
        Returns the MAP estimate of f(X).
        """
        mean, covariance = self.get_distribution(X)
        std_dev = np.zeros(mean.shape)
        for i in range(len(X)):
            std_dev[i] = np.sqrt(np.abs(self.get_distribution([X[i]])[1].item(0)))
        return mean, std_dev, covariance

    def sample(self, X):
        """
        Returns a sampled function f*(X).
        """
        mean, covariance = self.get_distribution(X)
        return np.random.multivariate_normal(mean, covariance)

    def _compute_cov(self, new_X):
        new_cov_mtx = self._compute_cov_mtx(new_X)
        if self.cov_mtx is None:
            self.cov_mtx = new_cov_mtx
        else:
            n = self.cov_mtx.shape[0]
            cov_ext = self._compute_cov_ext(new_X)
            self.cov_mtx = np.hstack((np.vstack((self.cov_mtx, cov_ext)), np.vstack((cov_ext.T, cov_mtx))))
    
    def _compute_cov_ext(self, new_X):
        X, Y = self.dataset
        n = len(X)
        new_n = len(new_X)
        cov_ext = np.zeros((new_n, n))
        for i in range(new_n):
            for j in range(n):
                cov_ext[i, j] = self.cov_func(new_X[i], X[j])
        return cov_ext
    
    def _compute_cov_mtx(self, X):
        n = len(X)
        cov_mtx = np.zeros((n, n)) 
        for i in range(n):
            for j in range(n):
                cov = self.cov_func(X[i], X[j])
                cov_mtx[i, j] = cov
                cov_mtx[j, i] = cov
        return cov_mtx


if __name__ == "__main__":
    main()

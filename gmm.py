# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-04-17 11:30 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import random


def gmm(X, K):
    threshold = 1e-7
    N, D = np.shape(X)
    centroids = np.array(random.sample(X, K))
    pMiu, pPi, pSigma = init_params(centroids, K, X, N, D)
    Lprev = -np.inf
    iters = 0
    while True:
        # Estiamtion Step
        Px = gaussian_prob(X, N, K, pMiu, pSigma, threshold, D)
        pGamma = Px * pPi
        pGamma = pGamma / np.sum(pGamma, axis=1, keepdims=True)
        # Maximization Step
        Nk = np.sum(pGamma, axis=0, keepdims=True)
        pMiu = pGamma.T.dot(X) / Nk.T
        pPi = Nk / N
        for kk in range(K):
            Xshift = X - pMiu[kk]
            pSigma[:, :, kk] = (Xshift.T * pGamma[:, kk].T).dot(Xshift) / Nk[0][kk]

        # check for convergence
        L = np.log(Px.dot(pPi.T)).sum()   # likelihood function
        print("iter {}: loss={:.6f}".format(iters, -L))
        iters+=1
        if L-Lprev < threshold:
            break
        Lprev = L

    return Px


def distance(X, centroids):
    """
    distance from data to every centroids
    return a N*K matrix.
    """
    Y = centroids
    n = len(X)
    m = len(Y)
    xx = np.sum(X*X, axis=1, keepdims=True)
    yy = np.sum(Y*Y, axis=1, keepdims=True)
    xy = X.dot(Y.T)
    return np.tile(xx, (1, m)) + np.tile(yy, (1, n)).T-2*xy


def init_params(centroids, K, X, N, D):
    """
    initiate parameters miu, pi, sigma:
        miu: random sample K points from data
        pi: percentage of points in every cluster
        sigma: covariance matrix of every cluster
    """

    pMiu = centroids
    pPi = np.zeros((1, K))
    pSigma = np.zeros((D, D, K))

    distmat = distance(X, centroids)
    labels = np.argmin(distmat, axis=1)

    for k in range(K):
        Xk = X[labels == k]
        pPi[0][k] = float(Xk.shape[0]) / N  
        pSigma[:, :, k] = np.cov(Xk.T)
    return pMiu, pPi, pSigma


def gaussian_prob(X, N, K, pMiu, pSigma, threshold, D):
    """
    compute N(X|mu_k, sigma_k)
    """
    Px = np.zeros((N, K))
    for k in range(K):
        Xshift = X - np.tile(pMiu[k], (N, 1))
        # add perturbation to covariance matrix in case of insufficient data
        inv_pSigma = inv(pSigma[:, :, k]) + threshold*np.identity(D) 
        tmp = np.sum(np.dot(Xshift, inv_pSigma) * Xshift, axis=1)
        coef = (2*np.pi)**(-D/2) * np.sqrt(np.linalg.det(inv_pSigma))
        Px[:, k] = coef * np.exp(-0.5 * tmp)
    return Px


def test():
    X = np.loadtxt('testSet.txt')
    ppx = gmm(X, 4)
    index = np.argmax(ppx, axis=1)
    plt.figure()
    plt.scatter(X[index == 0][:, 0], X[index == 0][:, 1], s=60, c=u'r', marker=u'o')
    plt.scatter(X[index == 1][:, 0], X[index == 1][:, 1], s=60, c=u'b', marker=u'o')
    plt.scatter(X[index == 2][:, 0], X[index == 2][:, 1], s=60, c=u'y', marker=u'o')
    plt.scatter(X[index == 3][:, 0], X[index == 3][:, 1], s=60, c=u'g', marker=u'o')
    plt.show()


if __name__ == '__main__':
    test()

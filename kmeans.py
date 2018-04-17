# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-04-16 17:01 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.

from __future__ import with_statement
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm
import random


def kmeans(X, k, observer=None, threshold=1e-5, maxiters=300):
    N = len(X)
    labels = np.zeros(N, dtype=np.int)
    centers = np.array(random.sample(X, k))
    iters = 0

    def Loss():
        L = 0
        for i in xrange(N):
            L += norm(X[i]-centers[labels[i]])
        return L

    def distance(X, Y):
        n = len(X)
        m = len(Y)
        xx = np.sum(X*X, axis=1, keepdims=True)  # 1450*1
        yy = np.sum(Y*Y, axis=1, keepdims=True)  # 3*1
        xy = np.dot(X, Y.T)                      # 1450*3
        #              1450*3              1450*3         1450*3
        return np.tile(xx, (1, m)) + np.tile(yy, (1, n)).T - 2*xy

    Jprev = Loss()
    while True:
        # notify the observer
        if observer is not None:
            observer(iters, labels, centers)

        # calculate distance from x to each center
        dist = distance(X, centers)
        # assign x to nearst center
        labels = dist.argmin(axis=1)
        # re-calculate each center
        for j in range(k):
            centers[j] = X[labels == j].mean(axis=0)

        J = Loss()
        print("iter {}: loss={:.4f}".format(iters, J))
        iters += 1

        if Jprev == J:
            print(" converged, stop. ")
            break
        if iters >= maxiters:
            print("reached max iterations, stop. ")
            break
        Jprev = J

    # final notification
    if observer is not None:
        observer(iters, labels, centers)


if __name__ == '__main__':
    with open('data/data.pkl') as inf:
        X = pickle.load(inf)

    def observer(iters, labels, centers):
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        plt.plot(hold=False)  # clear previous plot

        # draw points
        data_colors = [colors[lbl] for lbl in labels]
        plt.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.5)
        # draw centers
        plt.scatter(centers[:, 0], centers[:, 1], s=200, c=colors)

        plt.savefig('data/iters_%02d.png' % iters, format='png')

    kmeans(X, 3, observer=observer)

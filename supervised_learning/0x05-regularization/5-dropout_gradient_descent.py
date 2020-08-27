#!/usr/bin/env python3
"""dropout_gradient_descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with
    Dropout regularization using gradient descent"""
    devsz = []
    m = np.shape(Y)
    devlz = devsz.append(cache['A' + str(L)] - Y)
    for l in range(L, 0, -1):
        AT = cache['A' + str(l - 1)].T
        WT = weights['W' + str(l)].T
        A = cache['A' + str(l - 1)]
        devg = (1 - A**2)
        devWx = np.matmul(devsz[L - l], AT) / m[1]
        devbx = np.sum(devsz[L - l], axis=1, keepdims=True) / m[1]
        if l != 1:
            reg_drop = devg * (cache['D' + str(l - 1)] / keep_prob)
            devzx = devsz.append(np.matmul(WT, devsz[L - l]) * reg_drop)
        weights['W' + str(l)] -= alpha * devWx
        weights['b' + str(l)] -= alpha * devbx

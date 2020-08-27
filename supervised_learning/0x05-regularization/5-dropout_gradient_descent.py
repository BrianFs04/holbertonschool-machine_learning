#!/usr/bin/env python3
"""dropout_gradient_descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with
    Dropout regularization using gradient descent"""
    devsz = []
    m = np.shape(Y)
    devlz = devsz.append(cache['A' + str(L)] - Y)
    for ls in range(L, 0, -1):
        AT = cache['A' + str(ls - 1)].T
        WT = weights['W' + str(ls)].T
        A = cache['A' + str(ls - 1)]
        devg = (1 - A**2)
        devWx = np.matmul(devsz[L - ls], AT) / m[1]
        devbx = np.sum(devsz[L - ls], axis=1, keepdims=True) / m[1]
        if ls != 1:
            reg_drop = devg * (cache['D' + str(ls - 1)] / keep_prob)
            devzx = devsz.append(np.matmul(WT, devsz[L - ls]) * reg_drop)
        weights['W' + str(ls)] -= alpha * devWx
        weights['b' + str(ls)] -= alpha * devbx

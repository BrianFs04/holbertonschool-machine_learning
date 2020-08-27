#!/usr/bin/env python3
"""l2_reg_gradient_descent"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using
    gradient descent with L2 regularization"""
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
        devzx = devsz.append(np.matmul(WT, devsz[L - l]) * devg)
        l2 = devWx + ((lambtha / m[1]) * weights['W' + str(l)])
        weights['W' + str(l)] -= alpha * l2
        weights['b' + str(l)] -= alpha * devbx

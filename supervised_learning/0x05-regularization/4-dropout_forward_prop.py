#!/usr/bin/env python3
"""softmax, tanh, dropout_forward_prop"""
import numpy as np


def softmax(x):
    """Returns softmax function"""
    return(np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True))


def tanh(x):
    """Returns tanh function"""
    return(np.tanh(x))


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X
    for ls in range(L):
        W = weights['W' + str(ls + 1)]
        A = cache['A' + str(ls)]
        B = weights['b' + str(ls + 1)]
        ZX = np.matmul(W, A) + B
        if ls != L - 1:
            cache['A' + str(ls + 1)] = tanh(ZX)
            cache['D' + str(ls + 1)] = np.random.binomial(1, keep_prob,
                                                          size=ZX.shape)
            cache['A' + str(ls + 1)] = np.multiply(cache['A' + str(ls + 1)],
                                                   cache['D' + str(ls + 1)])
            cache['A' + str(ls + 1)] /= keep_prob
        else:
            cache['A' + str(L)] = softmax(ZX)
    return(cache)

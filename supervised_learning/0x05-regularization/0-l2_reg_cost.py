#!/usr/bin/env python3
"""l2_reg_cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization"""
    frobenius = 0
    for i in range(1, L + 1):
        key = 'W' + str(i)
        frobenius += np.linalg.norm(weights[key])
    reg_term = ((lambtha / (2 * m)) * frobenius)
    l2_cost = cost + reg_term
    return(l2_cost)

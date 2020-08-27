#!/usr/bin/env python3
"""l2_reg_cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization"""
    frobenius = 0
    for v in weights.values():
        frobenius += np.linalg.norm(v)
    reg_term = ((lambtha / (2 * m)) * frobenius)
    l2_cost = cost + reg_term
    return(l2_cost)

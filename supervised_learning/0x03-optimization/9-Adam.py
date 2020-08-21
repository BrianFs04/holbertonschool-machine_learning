#!/usr/bin/env python3
"""update_variables_Adam"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Adam optimization algorithm"""
    vd = (beta1 * v) + ((1 - beta1) * grad)
    sd = (beta2 * s) + ((1 - beta2) * (grad**2))
    tvd = vd / (1 - (beta1 ** t))
    tsd = sd / (1 - (beta2 ** t))
    var -= alpha * (tvd / ((tsd ** 0.5) + epsilon))
    return(var, vd, sd)

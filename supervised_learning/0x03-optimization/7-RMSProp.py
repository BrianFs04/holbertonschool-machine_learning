#!/usr/bin/env python3
"""update_variable_RMSProp"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """RMSProp optimization algorithm"""
    sd = (beta2 * s) + ((1 - beta2) * (grad**2))
    var -= alpha * (grad / ((sd ** 0.5) + epsilon))
    return(var, sd)

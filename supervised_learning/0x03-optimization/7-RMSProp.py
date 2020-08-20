#!/usr/bin/env python3
"""update_variables_RMSProp"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """RMSProp optimization algorithm"""
    sd = (beta2 * s) + ((1 - beta2) * grad**2)
    var -= alpha * (grad / np.sqrt(sd + epsilon))
    return(var, sd)

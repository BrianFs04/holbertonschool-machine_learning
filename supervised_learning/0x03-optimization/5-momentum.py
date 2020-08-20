#!/usr/bin/env python3
"""update_variable_momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient descent
    with momentum optimization algorithm"""
    vd = (beta1 * v) +  ((1- beta1) * grad)
    var -= alpha * vd
    return(var, vd)

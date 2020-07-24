#!/usr/bin/env python3
"""Func poly_derivate"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polynomial"""
    res = []
    if type(poly) is not list:
        return None
    if len(poly) == 1:
        return([0])
    for i in range(1, len(poly)):
        if type(poly[i]) is not int:
            return None
        res.append(poly[i] * i)
    return(res)

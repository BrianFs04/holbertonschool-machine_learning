#!/usr/bin/env python3
"""Func poly_derivate"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polynomial"""
    res = []
    for i in range(1, len(poly)):
        res.append(poly[i] * i)
    return(res)

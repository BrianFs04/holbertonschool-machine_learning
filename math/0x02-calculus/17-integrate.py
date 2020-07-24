#!/usr/bin/env python3
"""Func poly_integral"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    res = [C]
    for i in range(len(poly)):
        if type(poly[i]) is not int:
            return None
        if (i == 0) or (poly[i] == 0):
            res.append(poly[i])
        else:
            inte = poly[i] / (i + 1)
            if inte % 1 == 0:
                res.append(int(inte))
            else:
                res.append(inte)
    return(res)

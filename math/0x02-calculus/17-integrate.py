#!/usr/bin/env python3
"""Func poly_integral"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial"""
    if type(poly) is not list or len(poly) == 0 or type(C) is not int:
        return None
    res = [C]
    for i in range(len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
        if sum(poly) == 0:
            continue
        if type(poly[i]) is float:
            res.append(poly[i] * 10 / ((i + 1) * 10))
        elif (i == 0) or (poly[i] == 0):
            res.append(poly[i])
        else:
            inte = poly[i] / (i + 1)
            if inte % 1 == 0:
                res.append(int(inte))
            else:
                res.append(inte)
    return(res)

#!/usr/bin/env python3
"""Func summation_i_squared(n)"""


def summation_i_squared(n):
    """Function that calculates a squared sum"""
    if n <= 0 or type(n) is float:
        return None
    squared_sum = (n * (n + 1)) * ((2*n) + 1) / 6
    return(int(squared_sum))

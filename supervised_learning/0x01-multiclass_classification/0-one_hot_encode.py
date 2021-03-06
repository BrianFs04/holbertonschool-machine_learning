#!/usr/bin/env python3
"""one_hot_encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix"""
    if type(Y) is not np.ndarray or type(classes) is not int:
        return None
    for i in Y:
        if i >= classes:
            return None
    shape = (classes, np.size(Y))
    one_hot = np.zeros(shape)
    cols = np.arange(np.size(Y))
    one_hot[Y, cols] = 1
    return(one_hot)

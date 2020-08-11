#!/usr/bin/env python3
"""one_hot_decode"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    if type(one_hot) is not np.ndarray:
        return None
    if not 0 in one_hot and not 1 in one_hot:
        return None
    labels = np.argmax(one_hot, axis=0)
    return(labels)

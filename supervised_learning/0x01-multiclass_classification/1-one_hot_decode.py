#!/usr/bin/env python3
"""one_hot_decode"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    if type(one_hot) is not np.ndarray:
        return None
    if 0 not in one_hot or 1 not in one_hot:
        return None
    labels = np.argmax(one_hot, axis=0)
    return(labels)

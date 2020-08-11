#!/usr/bin/env python3
"""one_hot_decode"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    classes = one_hot.shape[0]
    m = one_hot.shape[1]
    if type(one_hot) is not np.ndarray:
        return None
    if m > classes:
        return None
    labels = np.argmax(one_hot, axis=0)
    return(labels)

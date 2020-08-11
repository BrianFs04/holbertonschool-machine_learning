#!/usr/bin/env python3
"""one_hot_decode"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    if type(one_hot) is not np.ndarray:
        return None
    for i in one_hot:
        if type(i) is not np.ndarray:
            return None
        for j in i:
            j = j.astype(np.int64)
            if j == 0 or j == 1:
                continue
            else:
                return None
    labels = np.argmax(one_hot, axis=0)
    return(labels)

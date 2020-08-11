#!/usr/bin/env python3
"""one_hot_decode"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    labels = np.argmax(one_hot, axis=0)
    return(labels)

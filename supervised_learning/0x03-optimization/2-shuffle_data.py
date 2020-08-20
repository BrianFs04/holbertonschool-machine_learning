#!/usr/bin/env python3
"""shuffle_data"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way"""
    s = np.random.permutation(X.shape[0])
    return (X[s], Y[s])

#!/usr/bin/env python3
"""Function np_cat"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenate numpy arrays"""
    new_mat = np.array([])
    new_mat = np.concatenate((mat1, mat2), axis)
    return(new_mat)

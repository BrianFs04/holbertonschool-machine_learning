#!/usr/bin/env python3
"""precision"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix"""
    tp = np.diagonal(confusion)
    fp = np.sum(confusion, axis=0)
    precision = tp / fp
    return(precision)

#!/usr/bin/env python3
"""sensitivity"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix"""
    tp = np.diagonal(confusion)
    p = np.sum(confusion, axis=1)
    return(tp / p)

#!/usr/bin/env python3
"""precision"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix"""
    TP = np.diagonal(confusion)
    FP = np.sum(confusion, axis=0) - TP
    PPV = TP / (TP + FP)
    return(PPV)

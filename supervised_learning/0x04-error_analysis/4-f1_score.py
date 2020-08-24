#!/usr/bin/env python3
"""f1_score"""
import numpy as np


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix"""
    tp = np.diagonal(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    f1 = 2 * tp / ((2 * tp) + fp + fn)
    return(f1)

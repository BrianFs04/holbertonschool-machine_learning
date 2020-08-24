#!/usr/bin/env python3
"""specificity"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix"""
    tp = np.diagonal(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = np.sum(confusion) - (fp + fn + tp)
    specificity = tn / (fp + tn)
    return(specificity)

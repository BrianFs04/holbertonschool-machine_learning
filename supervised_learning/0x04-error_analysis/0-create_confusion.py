#!/usr/bin/env python3
"""create_confusion_matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix"""
    c_matrix = np.matmul(labels.T, logits)
    return(c_matrix)

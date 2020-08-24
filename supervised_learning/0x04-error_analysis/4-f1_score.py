#!/usr/bin/env python3
"""f1_score"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix"""
    PPV = precision(confusion)
    TPR = sensitivity(confusion)
    f1 = 2 * ((PPV * TPR) / (PPV + TPR))
    return(f1)

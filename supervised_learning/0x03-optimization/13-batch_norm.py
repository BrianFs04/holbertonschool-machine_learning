#!/usr/bin/env python3
"""batch_norm"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural
    network using batch normalization"""
    m = Z.shape[0]
    mean = np.sum(Z, axis=0) / m
    variance = np.sum((Z - mean) ** 2, axis=0) / m
    sd = np.sqrt(variance)
    z_norm = (Z - mean) / (sd + epsilon)
    z_tilde = gamma * z_norm + beta
    return(z_tilde)

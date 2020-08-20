#!/usr/bin/env python3
import numpy as np
"""Normalization_constants"""


def normalization_constants(X):
  """Calculates the normalization (standardization) constants of a matrix"""
  length = X.shape[0]
  mean = np.sum(X, axis=0) / length
  variance = np.sum((X - mean) ** 2, axis=0) / length
  sd = np.sqrt(variance) 
  return(mean, sd)


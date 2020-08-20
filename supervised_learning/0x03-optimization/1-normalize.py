#!/usr/bin/env python3
"""normalize"""
import numpy as np


def normalize(X, m, s):
  """Normalizes (standardizes) a matrix"""
  Z = (X - m) / s
  return(Z)

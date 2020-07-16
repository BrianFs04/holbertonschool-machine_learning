#!/usr/bin/env python3
"""Adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Function that adds two matrices element-wise"""
    if len(mat1) == len(mat2):
        addi = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
                for i in range(len(mat1))]
        return(addi)
    else:
        return(None)

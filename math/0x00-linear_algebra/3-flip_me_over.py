#!/usr/bin/env python3
"""Transpose of a matrix"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix"""
    new_matrix = [[matrix[j][i] for j in range(len(matrix))]
                  for i in range(len(matrix[0]))]
    return(new_matrix)

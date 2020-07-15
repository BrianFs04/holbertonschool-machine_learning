#!/usr/bin/env python3
"""Size of a matrix"""


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix"""
    size = []
    size.append(len(matrix))
    if type(matrix[0]) == list:
        while type(matrix[0]) != int:
            size.append(len(matrix[0]))
            matrix = matrix[0]
    return(size)

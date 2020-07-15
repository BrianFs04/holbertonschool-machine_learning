#!/usr/bin/env python3
"""Adds two matrices element-wise"""


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix"""
    size = []
    size.append(len(matrix))
    if type(matrix[0]) == list:
        while type(matrix[0]) != int and type(matrix[0]) != float:
            size.append(len(matrix[0]))
            matrix = matrix[0]
    return(size)


def add_matrices2D(mat1, mat2):
    """Function that adds two matrices element-wise"""
    if matrix_shape(mat1) == matrix_shape(mat2):
        addi = [[mat1[i][j] + mat2[i][j] for j in range(len(mat2))]
                for i in range(len(mat1))]
        return(addi)
    else:
        return(None)

#!/usr/bin/env python3
"""Functions: matrix_shape
              add_matrices"""


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix"""
    size = []
    size.append(len(matrix))
    if type(matrix[0]) == list:
        while type(matrix[0]) != int:
            size.append(len(matrix[0]))
            matrix = matrix[0]
    return(size)


def add_matrices(mat1, mat2):
    """Function that adds two matrices"""
    if(type(mat1[0]) == int) and (len(mat1) == len(mat2)):
        return([mat1[i] + mat2[i] for i in range(len(mat1))])

    if(matrix_shape(mat1) == matrix_shape(mat2)):
        if(len(matrix_shape(mat1)) == 4):
            return([[[[mat1[i][j][k][l] + mat2[i][j][k][l]
                    for l in range(len(mat1[i][j][k]))]
                    for k in range(len(mat1[i][j]))]
                    for j in range(len(mat1[i]))]
                    for i in range(len(mat1))])
        elif(len(matrix_shape(mat1)) == 2):
            return([[mat1[i][j] + mat2[i][j] for j in range(len(mat2[0]))]
                    for i in range(len(mat1))])
    else:
        return(None)

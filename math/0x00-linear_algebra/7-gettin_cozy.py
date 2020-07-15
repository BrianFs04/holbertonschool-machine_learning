#!/usr/bin/env python3
"""Function cat_matrices2D"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis"""
    new_mat = []
    cop_mat1 = [[mat1[rows][cols] for cols in range(len(mat1[0]))]
                for rows in range(len(mat1))]
    cop_mat2 = [[mat2[rows][cols] for cols in range(len(mat2[0]))]
                for rows in range(len(mat2))]
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        new_mat = cop_mat1 + cop_mat2
        return(new_mat)
    if axis == 1 and len(mat1) == len(mat2):
        for i in range(len(cop_mat2)):
            new_mat += [cop_mat1[i] + cop_mat2[i]]
        return(new_mat)
    return(None)

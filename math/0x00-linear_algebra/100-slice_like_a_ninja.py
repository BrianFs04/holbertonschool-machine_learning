#!/usr/bin/env python3
"""Function: np_slice"""


def np_slice(matrix, axes={}):
    """Slices a matrix along a specific axes"""
    ax_vals = {}
    max_axis = max(axes.keys())

    for j in range(max_axis):
        ax_vals[j] = (None, None, None)

    for key in axes.keys():
        ax_vals[key] = axes[key]

    return(matrix[tuple(slice(*ax_vals[i]) for i in ax_vals)])

#!/usr/bin/env python3
"""convolve_grayscale_same"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    # Number of images, input height and input width
    m, h, w = images.shape
    # Filter height and filter width
    kh, kw = kernel.shape

    # Calculates padding height and width
    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)

    # Convolution output
    output = np.zeros((m, h, w))

    # pads images 0, top and bottom, left and right respectively
    images_padded = np.pad(images, [(0, 0), (ph, ph), (pw, pw)], 'constant')

    # Loop over every pixel of the output
    for i in range(h):
        for j in range(w):
            # Creating matrices 3x3 in accordance with the stride
            image = images_padded[:, i:i+kh, j:j+kw]
            # Element-wise multiplication of the kernel and the image
            res = kernel * image
            # Numpy addition in rows
            res = np.sum(res, axis=1)
            res = np.sum(res, axis=1)
            output[:, i, j] = res
    return(output)

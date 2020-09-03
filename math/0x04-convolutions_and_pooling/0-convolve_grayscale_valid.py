#!/usr/bin/env python3
"""convolve_grayscale_valid"""
import numpy as np
from math import ceil, floor


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale image"""
    # strides for height and width
    strides = (1, 1)
    # number of images, input height and input width
    m, h, w = images.shape
    # filter height and filter width
    kh, kw = kernel.shape

    # outputs when the padding is "valid"
    output_h = int(floor(float(h - kh + 1) / float(strides[0])))
    output_w = int(floor(float(w - kw + 1) / float(strides[1])))

    # convolution output
    output = np.zeros((m, output_h, output_w))

    # Loop over every pixel of the output
    for i in range(output_h):
        for j in range(output_w):
            # creating matrices 3x3
            image = images[:, i:i+kh, j:j+kw]
            # element-wise multiplication of the kernel and the image
            res = kernel * image
            # numpy addition in rows
            res = np.sum(res, axis=1)
            res = np.sum(res, axis=1)
            output[:, i, j] = res
    return(output)

#!/usr/bin/env python3
"""convolve_grayscale_same"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    # number of images, input height and input width
    m, h, w = images.shape
    # filter height and filter width
    kh, kw = kernel.shape

    # Calculate the number of zeros which are needed to add as padding
    ph = max((h - 1) + kh - h, 0)
    pw = max((w - 1) + kw - w, 0)

    # convolution output
    output = np.zeros((m, h, w))

    # pads images 0, top and bottom, left and right respectively
    images_padded = np.pad(images, [(0, 0), (ph, ph), (pw, pw)], 'constant')

    # Loop over every pixel of the output
    for i in range(h):
        for j in range(w):
            #creating matrices 3x3 in accordance with the stride
            image = images_padded[:, i:i+kh, j:j+kw]
            # element-wise multiplication of the kernel and the image
            res = kernel * image
            # numpy addition in rows
            res = np.sum(res, axis=1)
            res = np.sum(res, axis=1)
            output[:, i, j] = res
    return(output)

#!/usr/bin/env python3
"""convolve_grayscale_same"""
import numpy as np
from math import ceil, floor


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    # strides for height and width
    strides = (1, 1)
    # number of images, input height and input width
    m, h, w = images.shape
    # filter height and filter width
    kh, kw = kernel.shape

    # outputs when the padding is "same"
    output_h = int(ceil(float(h) / float(strides[0])))
    output_w = int(ceil(float(w) / float(strides[1])))

    # Calculate the number of zeros which are needed to add as padding
    ph = max((output_h - 1) * strides[0] + kh - h, 0)
    pw = max((output_w - 1) * strides[1] + kw - w, 0)

    # convolution output
    output = np.zeros((m, output_h, output_w))

    # pads images 0, top and bottom, left and right respectively
    images_padded = np.pad(images, [(0, 0), (ph, ph), (pw, pw)], 'constant')

    # Loop over every pixel of the output
    for i in range(output_h):
        for j in range(output_w):
            # creating matrices 3x3 in accordance with the stride
            image = images_padded[:, i * strides[0]:i * strides[0] + kh, j *
                                  strides[1]:j * strides[1] + kw]
            # element-wise multiplication of the kernel and the image
            res = kernel * image
            # numpy addition in rows
            res = np.sum(res, axis=1)
            res = np.sum(res, axis=1)
            output[:, i, j] = res

    return(output)

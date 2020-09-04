#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    # number of images, input height and input width
    m, h, w = images.shape
    # filter height and filter width
    kh, kw = kernel.shape
    # padding for height and padding for width
    ph, pw = padding

    # outputs
    output_h = h - kh + (2 * ph) + 1
    output_w = w - kw + (2 * pw) + 1

    # convolution output
    output = np.zeros((m, output_h, output_w))

    # pads images 0, top and bottom, left and right respectively
    images_padded = np.pad(images, [(0, 0), (ph, ph), (pw, pw)], 'constant')

    # Loop over every pixel of the output
    for i in range(output_h):
        for j in range(output_w):
            # creating matrices 3x3
            image = images_padded[:, i:i+kh, j:j+kw]
            # element-wise multiplication of the kernel and the image
            res = kernel * image
            # numpy addition in rows
            res = np.sum(res, axis=1)
            res = np.sum(res, axis=1)
            output[:, i, j] = res

    return(output)

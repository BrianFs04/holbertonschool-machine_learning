#!/usr/bin/env python3
"""pool_forward"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer of a neural network"""
    # strides for height and width
    sh, sw = stride
    # number of examples, height of the previous layer,
    # width of the previous layer, number of channels in the previous layer
    m, h_prev, w_prev, c_prev = A_prev.shape
    # filter height, filter width
    kh, kw, = kernel_shape

    # outputs
    output_h = int(float(h_prev - kh) / float(sh)) + 1
    output_w = int(float(w_prev - kw) / float(sw)) + 1

    # convolution output
    output = np.zeros((m, output_h, output_w, c_prev))

    # Loop over every pixel of the output
    for x in range(output_w):
        for y in range(output_h):
            # creating matrices 3x3x1
            image = A_prev[:, y * sh:y * sh + kh, x * sw:x * sw + kw]
            # pooling
            if mode == 'max':
                res = np.max(image, axis=(1, 2))
            elif mode == 'avg':
                res = np.mean(image, axis=(1, 2))
            output[:, y, x, :] = res
    return(output)

#!/usr/bin/env python3
"""pool_backward"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer of a neural network"""
    # strides for height and width
    sh, sw = stride
    # number of examples, height of the previous layer,
    # width of the previous layer, number of channels
    m, h_prev, w_prev, c = A_prev.shape
    # filter height, filter width
    kh, kw = kernel_shape

    # outputs
    output_h = int(float(h_prev - kh) / float(sh)) + 1
    output_w = int(float(w_prev - kw) / float(sw)) + 1

    dX = np.zeros(A_prev.shape)

    # Loop over every pixel of the output
    for i in range(m):
        for y in range(output_h):
            for x in range(output_w):
                for ch in range(c):
                    A = dA[i, y, x, ch]
                    # pooling
                    if mode == 'max':
                        res = np.max(A)
                        dX[i, y * sh:y * sh + kh,
                           x * sw:x * sw + kw, ch] += res
                    elif mode == 'avg':
                        res = np.mean(A)
                        dX[i, y * sh:y * sh + kh,
                           x * sw:x * sw + kw, ch] += res
    return(dX)

#!/usr/bin/env python3
"""pool function"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images"""
    # strides for height and width
    sh, sw = stride
    # number of images, input height, input width and channels
    m, h, w, c = images.shape
    # filter height and filter width
    kh, kw = kernel_shape

    # outputs for height and width
    output_h = int(float(h - kh) / float(sh)) + 1
    output_w = int(float(w - kw) / float(sw)) + 1

    # convolution output
    output = np.zeros((m, output_h, output_w, c))

    # Loop over every pixel of the output
    for i in range(output_h):
        for j in range(output_w):
            # creating matrices 3x3
            image = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            # max or avg pooling
            if mode == 'max':
                res = np.max(image, axis=(1, 2))
            elif mode == 'avg':
                res = np.mean(image, axis=(1, 2))
            output[:, i, j, :] = res
    return(output)

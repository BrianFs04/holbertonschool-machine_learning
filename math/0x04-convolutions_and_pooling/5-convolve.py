#!/usr/bin/env python3
"""convolve"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels"""
    # strides for height and width
    sh, sw = stride
    # number of images, input height, input width and channels
    m, h, w, c = images.shape
    # filter height, filter width, filter channels and number of kernels
    kh, kw, c, nc = kernels.shape

    if padding == 'same':
        # Calculate the number of zeros which are needed to add as padding
        ph = max((h - 1) * sh + kh - h, 0)
        pw = max((w - 1) * sw + kw - w, 0)
        ph = -(-ph // 2)
        pw = -(-pw // 2)
    elif padding == 'valid':
        # paddings to all directions equal to 0
        ph, pw = 0, 0
    else:
        ph, pw = padding

    output_h = int(float(h - kh + (2 * ph)) / float(sh)) + 1
    output_w = int(float(w - kw + (2 * pw)) / float(sw)) + 1

    # convolution output
    output = np.zeros((m, output_h, output_w, nc))

    # pads images 0, top and bottom, left and right respectively
    images_padded = np.pad(images, [(0, 0), (ph, ph), (pw, pw), (0, 0)],
                           'constant')

    # Loop over every pixel of the output
    for ch in range(nc):
        for x in range(output_w):
            for y in range(output_h):
                # creating matrices 3x3x3
                image = images_padded[:, y * sh:y * sh + kh,
                                      x * sw:x * sw + kw]
                # element-wise multiplication of the kernel and the image
                res = kernels[..., ch] * image
                # numpy addition in rows
                res = np.sum(res, axis=1)
                res = np.sum(res, axis=1)
                res = np.sum(res, axis=1)
                output[:, y, x, ch] = res
    return(output)

#!/usr/bin/env python3
import numpy as np
from math import ceil, floor


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    # strides for height and width
    sh, sw = stride
    # number of images, input height and input width
    m, h, w = images.shape
    # filter height and filter width
    kh, kw = kernel.shape

    if padding == 'same':
        # Calculate the number of zeros which are needed to add as padding
        ph = max((h - 1) * sh + kh - h, 0)
        pw = max((w - 1) * sw + kw - w, 0)
        ph = ceil(ph / 2)
        pw = ceil(pw / 2)
    elif padding == 'valid':
        # paddings to all directions equal to 0
        ph, pw = 0, 0
    else:
        ph, pw = padding

    output_h = int(float(h - kh + (2 * ph) + 1) / float(sh))
    output_w = int(float(w - kw + (2 * pw) + 1) / float(sw))

    # convolution output
    output = np.zeros((m, output_h, output_w))

    # pads images 0, top and bottom, left and right respectively
    images_padded = np.pad(images, [(0, 0), (ph, ph), (pw, pw)], 'constant')

    # Loop over every pixel of the output
    for i in range(output_h):
        for j in range(output_w):
            # creating matrices 3x3
            image = images_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            # element-wise multiplication of the kernel and the image
            res = kernel * image
            # numpy addition in rows
            res = np.sum(res, axis=1)
            res = np.sum(res, axis=1)
            output[:, i, j] = res

    return(output)

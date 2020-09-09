#!/usr/bin/env python3
"""conv_forward"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional
    layer of a neural network"""
    # strides for height and width
    sh, sw = stride
    # number of examples, height of the previous layer,
    # width of the previous layer, number of channels in the previous layer
    m, h_prev, w_prev, c_prev = A_prev.shape
    # filter height, filter width, numbers of channels
    # in the previoius layer and number of channels in the input
    kh, kw, c_prev, c_new = W.shape

    if padding == 'same':
        # Calculate the number of zeros which are needed to add as padding
        ph = max((h_prev - 1) * sh + kh - h_prev, 0)
        pw = max((w_prev - 1) * sw + kw - w_prev, 0)
        ph = -(-ph // 2)
        pw = -(-pw // 2)
    elif padding == 'valid':
        # paddings to all directions equal to 0
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # outputs
    output_h = int(float(h_prev - kh + (2 * ph)) / float(sh)) + 1
    output_w = int(float(w_prev - kw + (2 * pw)) / float(sw)) + 1

    # convolution output
    output = np.zeros((m, output_h, output_w, c_new))

    # pads images 0, top and bottom, left and right respectively
    images_padded = np.pad(A_prev, [(0, 0), (ph, ph),
                                    (pw, pw), (0, 0)], 'constant')

    # Loop over every pixel of the output
    for ch in range(c_new):
        for y in range(output_h):
            for x in range(output_w):
                # creating matrices 3x3x1
                image = images_padded[:, y * sh:y * sh + kh,
                                      x * sw:x * sw + kw]
                # element-wise multiplication of the kernel and the image
                res = W[..., ch] * image
                # numpy addition in rows adding the bias
                res = np.sum(res, axis=(1, 2, 3)) + b[..., ch]
                output[:, y, x, ch] = res
    return(activation(output))

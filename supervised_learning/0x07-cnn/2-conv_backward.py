#!/usr/bin/env python3
"""conv_backward"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional
    layer of a neural network"""
    # strides for height and width
    sh, sw = stride
    # number of examples, height of the previous layer,
    # width of the previous layer, number of channels in the previous layer
    m, h_prev, w_prev, c_prev = A_prev.shape
    # filter height, filter width
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

    # pads images 0, top and bottom, left and right respectively
    images_padded = np.pad(A_prev, [(0, 0), (ph, ph), (pw, pw),
                                    (0, 0)], 'constant')

    dX = np.zeros(images_padded.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Loop over every pixel of the output
    for i in range(m):
        for y in range(output_h):
            for x in range(output_w):
                for ch in range(c_new):
                    Z = dZ[i, y, x, ch]
                    # partial derivatives with respect to the previous layer
                    dX[i, y * sh:y * sh + kh,
                       x * sw:x * sw + kw, :] += Z * W[..., ch]
                    # kernels
                    dW[..., ch] += images_padded[i, y * sh:y * sh + kh,
                                                 x * sw:x * sw + kw, :] * Z
    if padding == 'same':
        dX = dX[:, ph:-ph, pw:-pw, :]
    return(dX, dW, db)

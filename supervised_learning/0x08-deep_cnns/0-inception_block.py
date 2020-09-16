#!/usr/bin/env python3
"""inception_block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters

    # First conv layer 1x1
    conv1 = K.layers.Conv2D(filters=F1, kernel_size=1,
                            activation="relu")(A_prev)

    # Conv layer 1x1 before the 3x3 conv layer
    conv2 = K.layers.Conv2D(filters=F3R, kernel_size=1, padding="same",
                            activation="relu")(A_prev)
    conv2_1 = K.layers.Conv2D(filters=F3, kernel_size=3, padding="same",
                              activation="relu")(conv2)

    # Conv layer 1x1 before the 5x5 conv layer
    conv3 = K.layers.Conv2D(filters=F5R, kernel_size=1, padding="same",
                            activation="relu")(A_prev)
    conv3_1 = K.layers.Conv2D(filters=F5, kernel_size=5, padding="same",
                              activation="relu")(conv3)

    # Max pooling layer before 1x1 conv layer
    max1 = K.layers.MaxPooling2D(pool_size=3, strides=1,
                                 padding="same")(A_prev)
    conv4 = K.layers.Conv2D(filters=FPP, kernel_size=1, padding="same",
                            activation="relu")(max1)

    output = K.layers.concatenate([conv1, conv2_1, conv3_1, conv4], axis=3)

    return(output)

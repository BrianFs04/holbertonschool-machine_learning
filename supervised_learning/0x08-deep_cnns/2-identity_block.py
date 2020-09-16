#!/usr/bin/env python3
"""identity_block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Builds an identity block"""
    F11, F3, F12 = filters

    # First component of main path
    conv1 = K.layers.Conv2D(filters=F11, kernel_size=1,
                            kernel_initializer="he_normal")(A_prev)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation("relu")(norm1)

    # Second component of main path
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=3,
                            kernel_initializer="he_normal",
                            padding="same")(act1)
    norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation("relu")(norm2)

    # Third component of main path
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=1,
                            kernel_initializer="he_normal",
                            padding="same")(act2)
    norm3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Shorcut value to main path, and pass it through ReLU activation
    X = K.layers.Add()([norm3, A_prev])
    output_act = K.layers.Activation("relu")(X)

    return(output_act)

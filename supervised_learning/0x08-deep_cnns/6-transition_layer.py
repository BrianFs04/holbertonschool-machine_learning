#!/usr/bin/env python3
"""transition_layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer"""
    norm1 = K.layers.BatchNormalization()(X)
    act1 = K.layers.Activation("relu")(norm1)
    nb_maps = int(compression * nb_filters)
    conv1 = K.layers.Conv2D(filters=nb_maps, kernel_size=1,
                            kernel_initializer="he_normal",
                            padding="same")(act1)
    avg1 = K.layers.AveragePooling2D(pool_size=2)(conv1)
    return(avg1, nb_maps)

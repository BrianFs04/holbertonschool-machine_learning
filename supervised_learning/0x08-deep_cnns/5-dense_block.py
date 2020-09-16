#!/usr/bin/env python3
"""dense_block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block"""
    concat_output = [X]

    for i in range(layers):
        norm1 = K.layers.BatchNormalization()(concat_output[i])
        act1 = K.layers.Activation("relu")(norm1)
        conv1 = K.layers.Conv2D(filters=128, kernel_size=1,
                                kernel_initializer="he_normal",
                                padding="same")(act1)
        norm2 = K.layers.BatchNormalization()(conv1)
        act2 = K.layers.Activation("relu")(norm2)
        conv2 = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                kernel_initializer="he_normal",
                                padding="same")(act2)
        output = K.layers.concatenate([concat_output[i], conv2])
        concat_output.append(output)
        nb_filters += growth_rate

    return(concat_output[-1], nb_filters)

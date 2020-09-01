#!/usr/bin/env python3
"""build_model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""
    L2 = K.regularizers.l2(lambtha)
    input = K.Input(shape=(nx, ))
    output = K.layers.Dense(layers[0], activations[0],
                            kernel_regularizer=L2)(input)
    for i in range(1, len(layers)):
        output = K.layers.Dropout(1 - keep_prob)(output)
        output = K.layers.Dense(layers[i], activations[i],
                                kernel_regularizer=L2)(output)
    model = K.Model(inputs=input, outputs=output)
    return(model)

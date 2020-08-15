#!/usr/bin/env python3
"""forward_prop"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network"""
    first = create_layer(x, layer_sizes[0], activations[0])
    for i in range(len(layer_sizes)):
        A = create_layer(first, layer_sizes[i], activations[i])
    return(A)

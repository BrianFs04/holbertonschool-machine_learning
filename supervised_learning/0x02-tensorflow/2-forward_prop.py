#!/usr/bin/env python3
"""forward_prop"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network"""
    for i in range(len(layer_sizes)):
        a = create_layer(x, layer_sizes[i], layer_sizes[i])
    return(a)

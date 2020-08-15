#!/usr/bin/env python3
"""forward_prop"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network"""
    if i is len(layer_sizes) - 1:
        a = create_layer(x, layer_sizes[i], tf.nn.softmax)
    a = create_layer(x, layer_sizes[i], activations[i])
    return(add_b)

#!/usr/bin/env python3
"""forward_prop"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network"""
    for ls in layer_sizes:
        for act in activations:
            a = create_layer(x, ls, act)
    return(a)

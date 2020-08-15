#!/usr/bin/env python3
"""forward_prop"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network"""
    b = tf.Variable(tf.zeros(layer_sizes[i]))
    a = create_layer(x, layer_sizes[i], activations[i])
    with tf.name_scope("layer"):
        add_b = tf.nn.bias_add(a, b)
        return(add_b)

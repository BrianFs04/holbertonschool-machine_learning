#!/usr/bin/env python3
"""create_layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Returns the tensor output of the layer"""
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    a = tf.layers.Dense(units=n, activation=activation, name='layer',
                        kernel_initializer=weights).apply(prev)
    return(a)

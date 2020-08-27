#!/usr/bin/env python3
"""dropout_create_layer"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout"""
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.contrib.layers.l2_regularizer(keep_prob)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=weights,
                            kernel_regularizer=dropout).apply(prev)
    return(layer)

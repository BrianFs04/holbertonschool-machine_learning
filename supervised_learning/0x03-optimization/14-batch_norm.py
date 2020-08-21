#!/usr/bin/env python3
"""create_batch_norm_layer"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in tensorflow"""

    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    z = tf.layers.Dense(n, activation, weights).apply(prev)

    mean, variance = tf.nn.moments(z, axes=[0])
    gamma = tf.Variable(1, dtype=tf.float32)
    beta = tf.Variable(0, dtype=tf.float32)
    epsilon = 1e-8

    batch_norm = tf.nn.batch_normalization(z, mean, variance,
                                           beta, gamma, epsilon)

    if not activation:
        return(batch_norm)
    return(activation(batch_norm))

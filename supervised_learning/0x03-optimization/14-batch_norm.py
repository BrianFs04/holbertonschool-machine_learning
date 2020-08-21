#!/usr/bin/env python3
"""crate_batch_norm_layer"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a
    neural network in tensorflow"""
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    z = tf.layers.Dense(units=n, kernel_initializer=weights).apply(prev)
    mean, variance = tf.nn.moments(z, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), True)
    epsilon = tf.constant(1e-8)
    batch_norm = tf.nn.batch_normalization(z, mean, variance,
                                           beta, gamma, epsilon)
    return(activation(batch_norm))

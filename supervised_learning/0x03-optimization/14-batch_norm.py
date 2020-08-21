#!/usr/bin/env python3
"""crate_batch_norm_layer"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a
    neural network in tensorflow"""
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    z = tf.layers.Dense(n, activation, weights).apply(prev)
    mean, variance = tf.nn.moments(z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), True, dtype=tf.float32)
    beta = tf.Variable(tf.zeros([n]), True, dtype=tf.float32)
    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(z, mean, variance,
                                           beta, gamma, epsilon)
    return(activation(batch_norm))

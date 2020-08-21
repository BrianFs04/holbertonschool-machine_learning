#!/usr/bin/env python3
"""create_Adam_op"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Adam optimization algorithm"""
    Adam = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
    return(Adam)

#!/usr/bin/env python3
"""create_RMSProp_op"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """RMSProp optimization algorithm"""
    RMSProp = tf.train.RMSPropOptimizer(alpha, beta2, 0.0,
                                        epsilon).minimize(loss)
    return(RMSProp)

#!/usr/bin/env python3
"""create_momentum_op"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Gradient descent with momentum optimization algorithm"""
    m_op = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return(m_op)

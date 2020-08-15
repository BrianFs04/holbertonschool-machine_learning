#!/usr/bin/env python3
"""create_placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """Returns two placeholders, x and y, for the neural network"""
    x = tf.placeholder(tf.float32, [None, nx], "x")
    y = tf.placeholder(tf.float32, [None, classes], "y")
    return(x, y)

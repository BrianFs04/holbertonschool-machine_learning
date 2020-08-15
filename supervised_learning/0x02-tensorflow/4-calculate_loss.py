#!/usr/bin/env python3
"""calculate_loss"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction"""
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return(loss)

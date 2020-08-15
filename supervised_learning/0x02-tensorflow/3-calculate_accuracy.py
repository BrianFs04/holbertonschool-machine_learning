#!/usr/bin/env python3
"""calculate_accuracy"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction"""
    prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return(accuracy)

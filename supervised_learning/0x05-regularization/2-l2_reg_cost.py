#!/usr/bin/env python3
"""l2_reg_cost"""
import tensorflow as tf


def l2_reg_cost(cost):
    """Calculates the cost of a neural network with L2 regularization"""
    l2_cost = tf.losses.get_regularization_losses(scope=None)
    return(cost + l2_cost)

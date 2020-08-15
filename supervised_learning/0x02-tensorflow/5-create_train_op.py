#!/usr/bin/env python3
"create_train_op"""
import tensorflow as tf


def create_train_op(loss, alpha):
  """Creates the training operation for the network"""
  gd = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
  return(gd)

#!/usr/bin/env python3
"""lenet5"""
import tensorflow as tf


def lenet5(x, y):
    """Builds a modified version of the
    LeNet-5 architecture using tensorflow"""
    # kernel initialized with he_normal method
    kernel = tf.contrib.layers.variance_scaling_initializer()

    # First convolutional layer
    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=5,
                             padding="same",
                             kernel_initializer=kernel,
                             activation=tf.nn.relu)(x)

    # First max pooling layer
    max1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    # Second convolutional layer
    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=5,
                             padding="valid",
                             kernel_initializer=kernel,
                             activation=tf.nn.relu)(max1)

    # Second max pooling layer
    max2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    # Flatten
    flat = tf.layers.Flatten()(max2)

    # Fully connected layers
    fc1 = tf.layers.Dense(units=120,
                          activation=tf.nn.relu,
                          kernel_initializer=kernel)(flat)
    fc2 = tf.layers.Dense(units=84,
                          activation=tf.nn.relu,
                          kernel_initializer=kernel)(fc1)
    fc3 = tf.layers.Dense(units=10,
                          kernel_initializer=kernel)(fc2)

    # softmax activated output
    output_act = tf.nn.softmax(fc3)

    # The accuracy of a prediction
    prediction = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    # The softmax cross-entropy loss of a prediction
    loss = tf.losses.softmax_cross_entropy(y, fc3)

    # Adam optimization algorithm
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return(output_act, optimizer, loss, accuracy)

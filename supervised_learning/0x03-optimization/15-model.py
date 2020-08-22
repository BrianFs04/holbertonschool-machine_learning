#!/usr/bin/env python3
"""Optimizated train"""
import numpy as np
import tensorflow as tf


def create_layer(prev, n, activation):
    """Returns the tensor output of the layer"""
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    a = tf.layers.Dense(units=n, activation=activation, name='layer',
                        kernel_initializer=weights).apply(prev)
    return(a)

def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a
    neural network in tensorflow"""
    if not activation:
      layer = create_layer(prev, n, activation)
      return(layer)

    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    z = tf.layers.Dense(units=n, kernel_initializer=weights).apply(prev)
    mean, variance = tf.nn.moments(z, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), True)
    epsilon = tf.constant(1e-8)
    batch_norm = tf.nn.batch_normalization(z, mean, variance,
                                           beta, gamma, epsilon)
    return(activation(batch_norm))

def create_placeholders(nx, classes):
    """Returns two placeholders, x and y, for the neural network"""
    x = tf.placeholder(tf.float32, [None, nx], "x")
    y = tf.placeholder(tf.float32, [None, classes], "y")
    return(x, y)

def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network"""
    A = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(activations)):
        A = create_batch_norm_layer(A, layer_sizes[i], activations[i])
    return(A)

def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction"""
    prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return(accuracy)

def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction"""
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return(loss)

def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way"""
    s = np.random.permutation(X.shape[0])
    return(X[s], Y[s])

def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Adam optimization algorithm"""
    Adam = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
    return(Adam)

def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates a learning rate decay operation"""
    rate_op = tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate, True)
    return(rate_op)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):

    x, y = create_placeholders(Data_train[0].shape[1], Data_train[1].shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    mini_batch = Data_train[0].shape[0] / batch_size
    if type(mini_batch) is not int:
        mini_batch = int(mini_batch) + 1

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epochs + 1):
            tc, ta = sess.run([loss, accuracy], feed_dict={x: Data_train[0],
                                                           y: Data_train[1]})
            vc, va = sess.run([loss, accuracy], feed_dict={x: Data_valid[0],
                                                           y: Data_valid[1]})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(tc))
            print("\tTraining Accuracy: {}".format(ta))
            print("\tValidation Cost: {}".format(vc))
            print("\tValidation Accuracy: {}".format(va))

            if i < epochs:
                xs, ys = shuffle_data(Data_train[0], Data_train[1])
                sess.run(global_step.assign(i))
                a = sess.run(alpha)
                for j in range(1, mini_batch + 1):
                    ft = (j - 1) * batch_size
                    lt = j * batch_size
                    if lt > Data_train[0].shape[0]:
                        lt = Data_train[0].shape[0]
                    batch = {x: xs[ft:lt], y: ys[ft:lt]}
                    sess.run(train_op, feed_dict=batch)
                    if j % 100 is 0:
                        cost = sess.run(loss, feed_dict=batch)
                        accur = sess.run(accuracy, feed_dict=batch)
                        print("\tStep {}:".format(j))
                        print("\t\tCost: {}".format(cost))
                        print("\t\tAccuracy: {}".format(accur))
        save_path = saver.save(sess, save_path)
    return(save_path)

#!/usr/bin/env python3
"""Optimizated train"""
import numpy as np
import tensorflow as tf

def shuffle_data(X, Y):
    """
    Returns: the shuffled X and Y matrices
    """
    vector = np.random.permutation(np.arange(X.shape[0]))
    X_shu = X[vector]
    Y_shu = Y[vector]
    return X_shu, Y_shu


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Returns: the Adam optimization operation
    """
    a = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                               beta2=beta2, epsilon=epsilon)
    optimize = a.minimize(loss)
    return optimize


def create_layer(prev, n, activation):
    """
    We have to use this function only in the last layer
    because we dont have to normalize the output
    Returns: the tensor output of the layer
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    A = tf.layers.Dense(units=n, name='layer', activation=activation,
                        kernel_initializer=init)
    Y_pred = A(prev)
    return (Y_pred)


def create_batch_norm_layer(prev, n, activation):
    """
    Returns: a tensor of the activated output for the layer
    """
    if activation is None:
        A = create_layer(prev, n, activation)
        return A

    # layers initialization
    w_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layers = tf.layers.Dense(units=n, kernel_initializer=w_init)
    Z = layers(prev)

    # trainable variables gamma and beta
    gamma = tf.Variable(tf.constant(1, dtype=tf.float32, shape=[n]),
                        name='gamma', trainable=True)
    beta = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n]),
                       name='beta', trainable=True)
    epsilon = tf.constant(1e-8)

    # Normalization Process
    mean, variance = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(x=Z, mean=mean, variance=variance,
                                       offset=beta, scale=gamma,
                                       variance_epsilon=epsilon)

    # activation of Z obtained
    A = activation(Z_norm)
    return A


def forward_prop(x, layers, activations):
    """
    forward propagation
    """
    A = create_batch_norm_layer(x, layers[0], activations[0])
    # hidden layers
    for i in range(1, len(activations)):
        A = create_batch_norm_layer(A, layers[i], activations[i])
    return A


def calculate_accuracy(y, y_pred):
    """
    accuracy of the prediction
    """
    index_y = tf.math.argmax(y, axis=1)
    index_pred = tf.math.argmax(y_pred, axis=1)
    comp = tf.math.equal(index_y, index_pred)
    cast = tf.cast(comp, dtype=tf.float32)
    accuracy = tf.math.reduce_mean(cast)
    return accuracy


def calculate_loss(y, y_pred):
    """
    loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Returns: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)



def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization"""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, False)
    global_increase = tf.assign(global_step, global_step + 1)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    mini_batch = X_train.shape[0] / batch_size
    if type(mini_batch) is not int:
        mini_batch = int(mini_batch)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(epochs + 1):
            tc, ta = sess.run([loss, accuracy], feed_dict={x: X_train,
                                                           y: Y_train})
            vc, va = sess.run([loss, accuracy], feed_dict={x: X_valid,
                                                           y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(tc))
            print("\tTraining Accuracy: {}".format(ta))
            print("\tValidation Cost: {}".format(vc))
            print("\tValidation Accuracy: {}".format(va))

            if i < epochs:
                xs, ys = shuffle_data(X_train, Y_train)
                for j in range(1, mini_batch + 1):
                    ft = (j - 1) * batch_size
                    lt = j * batch_size
                    if lt > X_train.shape[0]:
                        lt = X_train.shape[0]
                    batch = {x: xs[ft:lt], y: ys[ft:lt]}
                    sess.run(train_op, feed_dict=batch)
                    if j % 100 is 0:
                        cost = sess.run(loss, feed_dict=batch)
                        accur = sess.run(accuracy, feed_dict=batch)
                        print("\tStep {}:".format(j))
                        print("\t\tCost: {}".format(cost))
                        print("\t\tAccuracy: {}".format(accur))
            sess.run(global_increase)
        path = saver.save(sess, save_path)
    return(path)

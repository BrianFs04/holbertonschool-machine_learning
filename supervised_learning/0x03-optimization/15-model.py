#!/usr/bin/env python3
"""Model optim"""
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
    for i in range(1, len(layer_sizes)):
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
    rate_op = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                          decay_rate, True)
    return(rate_op)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization"""
    mini_iter = Data_train[0].shape[0] / batch_size
    if (mini_iter).is_integer() is True:
        mini_iter = int(mini_iter)
    else:
        mini_iter = int(mini_iter) + 1

    # building model
    x = tf.placeholder(tf.float32, shape=[None, Data_train[0].shape[1]],
                       name='x')
    tf.add_to_collection('x', x)
    y = tf.placeholder(tf.float32, shape=[None, Data_train[1].shape[1]],
                       name='y')
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    # Adam training & learning decay
    global_step = tf.Variable(0, trainable=False, name='global_step')

    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as ses:
        ses.run(init)
        # global initialization
        train_feed = {x: Data_train[0], y: Data_train[1]}
        valid_feed = {x: Data_valid[0], y: Data_valid[1]}

        for i in range(epochs + 1):
            T_cost = ses.run(loss, train_feed)
            T_acc = ses.run(accuracy, train_feed)
            V_cost = ses.run(loss, valid_feed)
            V_acc = ses.run(accuracy, valid_feed)
            print("After {} epochs:".format(i))
            print('\tTraining Cost: {}'.format(T_cost))
            print('\tTraining Accuracy: {}'.format(T_acc))
            print('\tValidation Cost: {}'.format(V_cost))
            print('\tValidation Accuracy: {}'.format(V_acc))

            if i < epochs:
                X_shu, Y_shu = shuffle_data(Data_train[0], Data_train[1])
                ses.run(global_step.assign(i))
                a = ses.run(alpha)

                for j in range(mini_iter):
                    ini = j * batch_size
                    fin = (j + 1) * batch_size
                    if fin > Data_train[0].shape[0]:
                        fin = Data_train[0].shape[0]
                    mini_feed = {x: X_shu[ini:fin], y: Y_shu[ini:fin]}

                    ses.run(train_op, feed_dict=mini_feed)
                    if j != 0 and (j + 1) % 100 == 0:
                        Min_cost = ses.run(loss, mini_feed)
                        Min_acc = ses.run(accuracy, mini_feed)
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(Min_cost))
                        print("\t\tAccuracy: {}".format(Min_acc))
        save_path = saver.save(ses, save_path)
    return save_path

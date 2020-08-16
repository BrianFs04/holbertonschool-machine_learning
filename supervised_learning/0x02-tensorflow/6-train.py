#!/usr/bin/env python3
"""train"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """Builds, trains, and saves a neural network classifier"""
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(iterations + 1):
            tc, ta = sess.run([loss, accuracy],
                              feed_dict={x: X_train, y: Y_train})
            vc, va = sess.run([loss, accuracy],
                              feed_dict={x: X_valid, y: Y_valid})
            if i % 100 is 0 or i is 0 or i is iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(tc))
                print("\tTraining Accuracy: {}".format(ta))
                print("\tValidation Cost: {}".format(vc))
                print("\tValidation Accuracy: {}".format(va))
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        path = saver.save(sess, save_path)
    return(path)

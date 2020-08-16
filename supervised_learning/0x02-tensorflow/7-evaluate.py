#!/usr/bin/env python3
"""evaluate"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network"""
    saver = tf.train.import_meta_graph(save_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        graph = tf.get_default_graph()
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]
        bp, res_accu, res_loss = sess.run([y_pred, accuracy, loss],
                                          feed_dict={x: X, y: Y})
    return(bp, res_accu, res_loss)

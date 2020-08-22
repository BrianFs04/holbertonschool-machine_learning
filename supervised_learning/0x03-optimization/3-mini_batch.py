#!/usr/bin/env python3
"""train_mini_batch"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a loaded neural network model
    using mini-batch gradient descent"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        mini_batch = X_train.shape[0] / batch_size
        if type(mini_batch) is not int:
            mini_batch = (int(mini_batch) + 1)

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
                for j in range(1, mini_batch):
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
        save_path = saver.save(sess, save_path)
    return(save_path)

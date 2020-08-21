#!/usr/bin/env python3
"""learning_rate_decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates a learning rate decay operation"""
    rate_op = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                          decay_rate, True)
    return(rate_op)

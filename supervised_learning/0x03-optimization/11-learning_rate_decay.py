#!/usr/bin/env python3
"""learning_rate_decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay"""
    alpha /= (1 + decay_rate * (global_step // decay_step))
    return(alpha)

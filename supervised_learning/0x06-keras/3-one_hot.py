#!/usr/bin/env python3
"""one_hot"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix"""
    one_hot = K.utils.to_categorical(labels, classes)
    return(one_hot)

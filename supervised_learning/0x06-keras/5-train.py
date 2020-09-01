#!/usr/bin/env python3
"""train_model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """Update the function to also analyze validaiton data"""
    history = network.fit(data, labels, batch_size, epochs, verbose,
                          shuffle=shuffle, validation_data=validation_data)
    return(history)

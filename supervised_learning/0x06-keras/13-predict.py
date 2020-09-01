#!/usr/bin/env python3
"""predict"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network"""
    prediction = network.predict(data, verbose=verbose)
    return(prediction)

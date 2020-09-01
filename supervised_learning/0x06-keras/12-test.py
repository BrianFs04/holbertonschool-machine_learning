#!/usr/bin/env python3
"""test_model"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network"""
    test = network.evaluate(data, labels, verbose=verbose)
    return(test)

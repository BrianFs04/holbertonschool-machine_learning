#!/usr/bin/env python3
"""save_config, load_config"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model’s configuration in JSON format"""
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)
    return(None)


def load_config(filename):
    """Loads a model with a specific configuration"""
    with open(filename, 'r') as f:
        loaded_config = K.models.model_from_json(f.read())
    return(loaded_config)

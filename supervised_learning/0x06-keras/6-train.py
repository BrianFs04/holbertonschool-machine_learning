#!/usr/bin/env python3
"""train_model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """Update the function to train the model using early stopping"""
    early_stop = None
    if validation_data:
        early_stop = [K.callbacks.EarlyStopping(patience=patience,
                                                monitor='val_loss')]
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=early_stop)
    return(history)

#!/usr/bin/env python3
"""train_model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True,
                shuffle=False):
    """Save the best iteration of the model"""
    early_stop = None
    if validation_data:
        early_stop = K.callbacks.EarlyStopping(patience=patience,
                                               monitor='val_loss')
        if learning_rate_decay:
            def lr_decay(step):
                """Updates the learning rate using inverse time decay"""
                return(alpha / (1 + decay_rate * step))
            lr = K.callbacks.LearningRateScheduler(schedule=lr_decay,
                                                   verbose=1)
    if filepath:
        save = K.callbacks.ModelCheckpoint(filepath,
                                           save_best_only=save_best
                                           mode='min')
    history = network.fit(x=data,
                          y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=[early_stop, lr, save])
    return(history)

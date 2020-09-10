#!/usr/bin/env python3
"""lenet5"""
import tensorflow.keras as K


def lenet5(X):
    """Builds a modified version of the LeNet-5 architecture using keras"""
    # First convolutional layer
    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=5,
                            padding="same",
                            kernel_initializer="he_normal",
                            activation="relu")(X)

    # First max pooling layer
    max1 = K.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    # Second convolutional layer
    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=5,
                            padding="valid",
                            kernel_initializer="he_normal",
                            activation="relu")(max1)

    # Second max pooling layer
    max2 = K.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    # Flatten
    flat = K.layers.Flatten()(max2)

    # Fully connected layers
    fc1 = K.layers.Dense(units=120,
                         activation="relu",
                         kernel_initializer="he_normal")(flat)
    fc2 = K.layers.Dense(units=84,
                         activation="relu",
                         kernel_initializer="he_normal")(fc1)
    fc3 = K.layers.Dense(units=10,
                         activation="softmax",
                         kernel_initializer="he_normal")(fc2)

    # Adam optimization algorithm
    Adam = K.optimizers.Adam()

    # Build the model
    model = K.Model(inputs=X, outputs=fc3)

    # Sets up Adam optimization for a keras model with categorical
    # crossentropy loss and accuracy metrics
    model.compile(Adam, "categorical_crossentropy", metrics=["accuracy"])

    return(model)

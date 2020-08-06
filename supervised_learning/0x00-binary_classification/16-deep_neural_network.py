#!/usr/bin/env python3
"""Class DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a neural defines a deep neural network
    performing binary classification"""

    def __init__(self, nx, layers):
        """Constructor method"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list:
            raise TypeError('layers must be a list of positive integers')
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for l in range(self.L):
            heetal = np.random.randn(layers[l], nx)*np.sqrt(2 / nx)
            if layers[l] < 1:
                raise ValueError('layers must be a list of positive integers')
            self.weights['W' + str(l + 1)] = heetal
            self.weights['b' + str(l + 1)] = np.zeros((layers[l], 1))
            nx = layers[l]

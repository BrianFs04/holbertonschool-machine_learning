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
        if type(layers) is not list or len(layers) is 0:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(self.__L):
            if type(layers[l]) is not int or layers[l] <= 0:
                raise TypeError('layers must be a list of positive integers')
            heetal = np.random.randn(layers[l], nx)*np.sqrt(2 / nx)
            self.__weights['W' + str(l + 1)] = heetal
            self.__weights['b' + str(l + 1)] = np.zeros((layers[l], 1))
            nx = layers[l]

    @property
    def L(self):
        """The number of layers in the neural network"""
        return(self.__L)

    @property
    def cache(self):
        """A dictionary to hold all intermediary values of the network"""
        return(self.__cache)

    @property
    def weights(self):
        """A dictionary to hold all weights and biased of the network"""
        return(self.__weights)

    def sigmoid(self, x):
        """Returns sigmoid function"""
        return(1/(1 + np.exp(-x)))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X
        for l in range(self.__L):
            w = self.__weights['W' + str(l + 1)]
            a = self.__cache['A' + str(l)]
            b = self.__weights['b' + str(l + 1)]
            zx = np.matmul(w, a) + b
            self.__cache['A' + str(l + 1)] = self.sigmoid(zx)
        return(self.__cache['A' + str(self.__L)], self.__cache)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = np.shape(Y)
        cost = -(1 / m[1]) * np.sum(Y * np.log(A) + (1 - Y) *
                                    np.log(1.0000001 - A))
        return(cost)

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

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A3, self.__cache = self.forward_prop(X)
        cont = np.where(A3 >= 0.5, 1, 0)
        return(cont, self.cost(Y, A3))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        devsz = []
        m = np.shape(Y)
        devlz = devsz.append(self.__cache['A' + str(self.__L)] - Y)
        for l in range(self.__L + 1, 1, -1):
            AT = self.__cache['A' + str(l - 2)].T
            WT = self.__weights['W' + str(l - 1)].T
            A = self.__cache['A' + str(l - 2)]
            devg = (A * (1 - A))
            devWx = np.matmul(devsz[4 - l], AT) / m[1]
            devbx = np.sum(devsz[4 - l], axis=1, keepdims=True) / m[1]
            devzx = devsz.append(np.matmul(WT, devsz[4 - l]) * devg)
            self.__weights['W' + str(l - 1)] -= alpha * devWx
            self.__weights['b' + str(l - 1)] -= alpha * devbx

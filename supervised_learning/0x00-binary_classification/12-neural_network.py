#!/usr/bin/env python3
"""Class NeuralNetwork"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden
    layer performing binary classification"""

    def __init__(self, nx, nodes):
        """Constructor method"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """The weights vector for the hidden layer"""
        return(self.__W1)

    @property
    def b1(self):
        """The bias for the hidden layer"""
        return(self.__b1)

    @property
    def A1(self):
        """The activated output for the hidden layer"""
        return(self.__A1)

    @property
    def W2(self):
        """The weights vector for the output neuron"""
        return(self.__W2)

    @property
    def b2(self):
        """The bias for the output neuron"""
        return(self.__b2)

    @property
    def A2(self):
        """The activated output for the output neuron (prediction)"""
        return(self.__A2)

    def sigmoid(self, x):
        """Returns sigmoid function"""
        return(1/(1 + np.exp(-x)))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(z1)
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(z2)
        return(self.__A1, self.__A2)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = np.shape(Y)
        cost = -(1 / m[1]) * np.sum(Y * np.log(A) + (1 - Y) *
                                    np.log(1.0000001 - A))
        return(cost)

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A1, A2 = self.forward_prop(X)
        cont = np.where(A2 >= 0.5, 1, 0)
        return(cont, self.cost(Y, A2))

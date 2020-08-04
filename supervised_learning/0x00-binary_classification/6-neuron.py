#!/usr/bin/env python3
"""Class Neuron"""
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Constructor method"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """The weights vector for the neuron"""
        return(self.__W)

    @property
    def b(self):
        """The bias for the neuron"""
        return(self.__b)

    @property
    def A(self):
        """The activated output of the neuron (prediction)"""
        return(self.__A)

    def sigmoid(self, x):
        """Returns sigmoid function"""
        return(1/(1 + np.exp(-x)))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = self.sigmoid(z)
        return(self.__A)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = np.shape(Y)
        cost = -(1 / m[1]) * np.sum(Y * np.log(A) + (1 - Y) *
                                    np.log(1.0000001 - A))
        return(cost)

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        predict = self.forward_prop(X)
        cond = np.where(predict >= 0.5, 1, 0)
        return(cond, self.cost(Y, predict))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = np.shape(Y)
        loss = A - Y
        b_dev = np.sum(loss) / m[1]
        gradient = np.matmul(loss, X.transpose()) / m[1]
        self.__W = self.__W - alpha * gradient
        self.__b = self.__b - alpha * b_dev

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron"""
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if not iterations > 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if not alpha > 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return(self.evaluate(X, Y))

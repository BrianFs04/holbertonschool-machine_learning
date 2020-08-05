#!/usr/bin/env python3
"""Class NeuralNetwork"""
import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = np.shape(Y)
        devz2 = A2 - Y
        devW2 = np.matmul(devz2, A1.T) / m[1]
        devb2 = np.sum(devz2, axis=1, keepdims=True) / m[1]
        devz1 = np.matmul(self.__W2.T, devz2) * (A1*(1 - A1))
        devW1 = np.matmul(devz1, X.T) / m[1]
        devb1 = np.sum(devz1, axis=1, keepdims=True) / m[1]
        self.__W1 = self.__W1 - alpha * devW1
        self.__b1 = self.__b1 - alpha * devb1
        self.__W2 = self.__W2 - alpha * devW2
        self.__b2 = self.__b2 - alpha * devb2

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network"""
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if not iterations > 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if not alpha > 0:
            raise ValueError('alpha must be positive')
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if not step > 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        x = []
        y = []
        for i in range(iterations + 1):
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            cost = self.cost(Y, self.__A2)
            if verbose is True:
                if i % step == 0:
                    print('Cost after {} iterations: {}'.format(i, cost))
                    x.append(i)
                    y.append(cost)
        if graph is True:
            plt.plot(x, y)
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()
        return(self.evaluate(X, Y))

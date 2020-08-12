#!/usr/bin/env python3
"""Class DeepNeuralNetwork"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork:
    """Defines a neural defines a deep neural network
    performing binary classification"""

    def __init__(self, nx, layers, activation='sig'):
        """Constructor method"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) is 0:
            raise TypeError('layers must be a list of positive integers')
        if activation is not 'sig' or activation is not 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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

    @property
    def activation(self):
        """Represents the type of activation
        function used in the hidden layers"""
        return(self.__activation)

    def sigmoid(self, x):
        """Returns sigmoid function"""
        return(1/(1 + np.exp(-x)))

    def softmax(self, x):
        """Returns softmax function"""
        return(np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True))

    def tahn(self, x):
        """Returns tahn function"""
        return(np.tahn(x))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X
        for ls in range(self.__L):
            w = self.__weights['W' + str(ls + 1)]
            a = self.__cache['A' + str(ls)]
            b = self.__weights['b' + str(ls + 1)]
            zx = np.matmul(w, a) + b
            if ls != self.__L - 1:
                if self.__activation is 'sig':
                    self.__cache['A' + str(ls + 1)] = self.sigmoid(zx)
                else:
                    self.__cache['A' + str(ls + 1)] = self.tahn(zx)
            else:
                self.__cache['A' + str(self.__L)] = self.softmax(zx)
        return(self.__cache['A' + str(self.__L)], self.__cache)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = len(A[0])
        cost = -np.sum((np.log(A) * Y) / m)
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
        for l in range(self.__L, 0, -1):
            AT = self.__cache['A' + str(l - 1)].T
            WT = self.__weights['W' + str(l)].T
            A = self.__cache['A' + str(l - 1)]
            if self.__activation is 'sig':
                devg = (A * (1 - A))
            if self.__activation is 'tahn':
                devg = (1 - (np.tanh(A))**2)
            devWx = np.matmul(devsz[self.__L - l], AT) / m[1]
            devbx = np.sum(devsz[self.__L - l], axis=1, keepdims=True) / m[1]
            devzx = devsz.append(np.matmul(WT, devsz[self.__L - l]) * devg)
            self.__weights['W' + str(l)] -= alpha * devWx
            self.__weights['b' + str(l)] -= alpha * devbx

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
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
        for i in range(iterations):
            A3, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            cost = self.cost(Y, A3)
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

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        ext = os.path.splitext(filename)[-1]
        if ext != '.pkl':
            filename = filename + '.pkl'
        with open(filename, 'wb') as fileObject:
            pickle.dump(self, fileObject)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        if not(os.path.exists(filename)):
            return None
        with open(filename, 'rb') as fileObject:
            x = pickle.load(fileObject)
            return(x)

#!/usr/bin/env python3
"""this module contains the a deep neural network class"""
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
        if activation is not 'sig' and activation is not 'tanh':
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

    def tanh(self, x):
        """Returns tanh function"""
        return(np.tanh(x))

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
                elif self.__activation is 'tanh':
                    self.__cache['A' + str(ls + 1)] = self.tanh(zx)
            else:
                self.__cache['A' + str(self.__L)] = self.softmax(zx)
        return(self.__cache['A' + str(self.__L)], self.__cache)

    def cost(self, Y, A):
        """calculates the cost of the model using logistic regression"""
        m = len(A[0])
        cost = -np.sum((np.log(A) * Y) / m)
        return cost

    def evaluate(self, X, Y):
        """evaluates the network's predictions"""
        A, cache = self.forward_prop(X)
        A_max = np.amax(A, axis=0)
        return np.where(A == A_max, 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ calculates one pass of the gradient descent on NN"""
        dZ = []
        m = np.shape(Y)[1]
        L = self.__L
        sa = self.__activation
        dZ.append(cache['A' + str(L)] - Y)
        for l in range(L, 0, -1):
            A = cache['A' + str(l - 1)]
            W = self.__weights['W' + str(l)]
            dg = (A * (1 - A)) if sa == 'sig' else 1 - A ** 2
            dWdx = np.matmul(dZ[L - l], A.T) / m
            dbdx = np.sum(dZ[L - l], axis=1, keepdims=True) / m
            dzdx = dZ.append(np.matmul(W.T, dZ[L - l]) * dg)
            self.__weights['W' + str(l)] -= alpha * dWdx
            self.__weights['b' + str(l)] -= alpha * dbdx

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """trains the DNN by updating the
        weights and cache private attributes"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        cost, iters = [], []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cst = self.cost(Y, A)
            if verbose is True:
                if i % step == 0:
                    print("Cost after {} iterations: {}".format(i, cst))
                    cost.append(cst)
                    iters.append(i)
        if graph is True:
            plt.plot(cost, iters)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ saves the instance of an object in pickle format"""
        f = open(filename, 'wb')
        pickle.dump(self, f)
        f.close()
        ext = os.path.splitext(filename)[-1].lower()
        if ext != '.pkl':
            os.rename(filename, filename + '.pkl')

    def load(filename):
        """ loads a pickled DeepNeuralNetwork object """
        if not filename:
            return None
        if os.path.exists(filename) is not True:
            return None
        else:
            with open(filename, 'rb') as f:
                pkl = pickle.load(f)
                return pkl

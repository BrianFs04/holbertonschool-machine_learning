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

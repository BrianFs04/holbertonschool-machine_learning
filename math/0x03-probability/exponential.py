#!/usr/bin/env python3
"""Class Exponential"""
e = 2.7182818285


class Exponential:
    """Represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Calculates the lambtha of data"""
        res = 0
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            for i in range(len(data)):
                res += data[i]
            self.lambtha = 1 / (res / len(data))

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""
        if x < 0:
            return 0
        pdf = self.lambtha* (e**(-self.lambtha*x))
        return(pdf)

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period"""
        if x < 0:
            return 0
        cdf = 1 - (e**(-self.lambtha*x))
        return(cdf)

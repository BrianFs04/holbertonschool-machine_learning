#!/usr/bin/env python3
"""Class Poisson"""
e = 2.7182818285


class Poisson:
    """Represents a poisson distribution"""

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
            self.lambtha = res / len(data)

    def factorial(self, val):
        """Calculates factorial of a number"""
        fact = val
        if val is 0:
            return 1
        for i in range(val - 1, 0, -1):
            fact *= i
        return(fact)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        pmf = (((self.lambtha**k)*(e**-self.lambtha)) / self.factorial(k))
        return(pmf)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        cdf = 0
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        for i in range(k + 1):
            cdf += (((e**-self.lambtha)*(self.lambtha**i)) / self.factorial(i))
        return(cdf)

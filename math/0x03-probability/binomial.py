#!/usr/bin/env python3
"""Class Binomial"""
e = 2.7182818285
π = 3.1415926536


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Calculates n and p from data"""
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p < 0 and p > 1:
                raise ValueError('p must be greater than 0 and less than 1')
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = var = 0

            for i in range(len(data)):
                mean += data[i]

            mean = mean / len(data)

            for i in range(len(data)):
                var += (data[i] - mean)**2

            var = var / len(data)

            s = 1 - (var / mean)
            self.n = round(mean / s)
            self.p = mean / self.n

    def factorial(self, val):
        """Calculates factorial of a number"""
        fact = val
        if val is 0:
            return 1
        for i in range(val - 1, 0, -1):
            fact *= i
        return(fact)

    def nCr(self, n, r):
        """Calculates the combination"""
        return(self.factorial(n) // (self.factorial(r) * self.factorial(n-r)))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        q = 1 - self.p
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        pmf = self.nCr(self.n, k) * (self.p**k) * (q**(self.n - k))
        return(pmf)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        cdf = 0
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return(cdf)

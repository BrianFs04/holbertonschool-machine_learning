#!/usr/bin/env python3
import math
"""Class Normal"""
e = 2.7182818285
π = 3.1415926536


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Calculates the mean and standard deviation of data"""
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev < 0:
                raise ValueError('stddev must be a positive value')
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            add = var = 0

            for i in range(len(data)):
                add += data[i]

            self.mean = add / len(data)

            for i in range(len(data)):
                var += (data[i] - self.mean)**2
            var = var / len(data)

            self.stddev = var**(1/2)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        zs = (x - self.mean) / self.stddev
        return(zs)

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        xv = (z * self.stddev) + self.mean
        return(xv)

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        m = 1/(self.stddev*(2*π)**(1/2))
        n = e**((-1/2)*((x - self.mean) / self.stddev)**2)
        pdf = m*n
        return(pdf)

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        m = (x - self.mean) / (self.stddev*2**(1/2))
        erf = 2/(π**(1/2))*(m - m**3/3 + m**5/10 - m**7/42 + m**9/216)
        cdf = 1/2*(1 + erf)
        return(cdf)

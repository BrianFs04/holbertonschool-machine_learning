#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))

poly = [0, 2, 0, 1]
print(poly_integral(poly))

poly = [5]
print(poly_integral(poly))

poly = [0, 1.7]
print(poly_integral(poly))

poly = [0, "1.7"]
print(poly_integral(poly))

poly = [0, 's']
print(poly_integral(poly))

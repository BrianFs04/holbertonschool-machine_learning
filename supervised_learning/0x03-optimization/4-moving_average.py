#!/usr/bin/env python3
"""moving_average"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set"""
    m_beta = 1 - beta
    vx = []
    res = []

    for i in range(len(data)):
        if i is 0:
            vx.append(0 + m_beta * data[i])
        else:
            vx.append(beta * vx[i - 1] + m_beta * data[i])

    for j in range(len(vx)):
        res.append(vx[j] / (1 - (beta ** (j + 1))))

    return(res)

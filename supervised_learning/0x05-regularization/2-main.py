#!/usr/bin/env python3

import tensorflow as tf
l2_reg_cost = __import__('2-l2_reg_cost').l2_reg_cost

if __name__ == '__main__':
    x = l2_reg_cost(tf.Variable(877))
    print(x)

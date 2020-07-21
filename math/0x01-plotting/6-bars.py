#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

labels = ['Farrah', 'Fred', 'Felicia']
xlen = np.arange(len(labels))
width = 0.5
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 81, 10))
plt.ylim(0, 80)
plt.xticks(xlen, labels)
apples = plt.bar(xlen, fruit[0], width, color='r')
bananas = plt.bar(xlen, fruit[1], width, bottom=fruit[0], color='yellow')
oranges = plt.bar(xlen, fruit[2], width, bottom=fruit[0]+fruit[1],
                  color='#ff8000')
peaches = plt.bar(xlen, fruit[3], width, bottom=fruit[0]+fruit[1]+fruit[2],
                  color='#ffe5b4')
plt.legend((apples, bananas, oranges, peaches),
           ('apples', 'bananas', 'oranges', 'peaches'))
plt.show()

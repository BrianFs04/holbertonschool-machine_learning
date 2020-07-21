#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig = plt.figure()
grid = plt.GridSpec(3, 2, hspace=1, wspace=0.5)
fig.suptitle('All in One')

fig0 = fig.add_subplot(grid[0, 0])
x = np.arange(0, 11)
fig0.set(xlim=(0, 10), xticks=(range(0, 11, 2)), yticks=(range(0, 1001, 500)))
fig0.plot(x, y0, color='r')

fig1 = fig.add_subplot(grid[0, 1])
fig1.scatter(x1, y1, color='magenta')
fig1.set_title("Men's Height vs Weight", size='x-small')
fig1.set_xlabel('Height (in)', size='x-small')
fig1.set_ylabel('Weight (lbs)', size='x-small')

fig2 = fig.add_subplot(grid[1, 0])
fig2.plot(x2, y2)
fig2.set(xlim=(0, 28651), yscale='log')
fig2.set_title('Exponential Decay of C-14', size='x-small')
fig2.set_xlabel('Time (years)', size='x-small')
fig2.set_ylabel('Fraction Remaining', size='x-small')

fig3 = fig.add_subplot(grid[1, 1])
fig3.set_title('Exponential Decay of Radioactive Elements', size='x-small')
lines = fig3.plot(x3, y31, 'r--', x3, y32, 'g')
fig3.set_xlabel('Time (years)', size='x-small')
fig3.set_ylabel('Fraction Remaining', size='x-small')
fig3.set(xlim=(0, 20000), ylim=(0, 1))
fig3.legend(lines[:2], ['C-14', 'Ra-226'], loc='upper right', prop={'size': 7})

fig4 = fig.add_subplot(grid[2, :])
fig4.set(ylim=(0, 30), xlim=(0, 100), xticks=(range(0, 101, 10)))
fig4.set_title('Project A', size='x-small')
fig4.set_xlabel('Grades', size='x-small')
fig4.set_ylabel('Number of students', size='x-small')
fig4.hist(student_grades, bins=range(0, 101, 10), edgecolor="black")

plt.show()

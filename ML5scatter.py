import numpy as np
import matplotlib.pyplot as plt

# x = [1,129,12,124,145,345,565,232,656,43]
# y = [98,76,34,32,65,61,23,43,54,90]

x = np.random.normal(0.0,5.0,1000)
y = np.random.normal(4.0,1.0,1000)

plt.scatter(x,y)
plt.show()
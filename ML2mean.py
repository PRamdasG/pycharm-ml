import numpy as np
from scipy import stats
speed = [12,43,5,423,43,32,434,5,7,78,67,66]

mean = np.mean(speed)
print("mean:",mean)
median = np.median(speed)
print("median:",median)
mode = stats.mode(speed)
print("mode:",mode)
std = np.std(speed)
print("std dev:",std)
var = np.var(speed)
print("variance:",var)
per = np.percentile(speed,90)
print("percentile:",per)

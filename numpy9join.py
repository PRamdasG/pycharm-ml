import numpy as np
arr1 = np.array([1,2,3,4,5,6,7,8])
arr2 = np.array([11,21,31,41,51,61,17,18])

newarr = np.concatenate((arr1,arr2))
print(newarr)
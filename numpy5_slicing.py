import numpy as np
arr = np.array([1,2,3,4,5,6,7,8])
print(arr[1:5])
print(arr[1:7:2])
print(arr[::-1])
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[1, 1:4])
print(arr.dtype)
arr1 = np.array(['apple', 'banana', 'cherry'])
print(arr1.dtype)

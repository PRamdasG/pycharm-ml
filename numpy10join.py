import numpy as np
# arr1 = np.array([[1,2],[4,5]])
# arr2 = np.array([[6,7],[8,9]])
#
# arr = np.concatenate((arr1,arr2),axis=1)

# arr1 = np.array([[1,2,4,5]])
# arr2 = np.array([[6,7,8,9]])
#
# arr = np.concatenate((arr1,arr2),axis=1)

arr1 = np.array([[1,2,4,5]])
arr2 = np.array([[6,7,8,9]])
# arr = np.hstack((arr1,arr2))
# arr = np.vstack((arr1,arr2))
arr = np.dstack((arr1,arr2))
print(arr)
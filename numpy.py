import numpy as np
# arr[행 인덱스, 열 인덱스]
# 5 행 / 3 열
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
print(arr)
print('shape :', arr.shape)
print()

# arr 0~2 행 / 3열
print('0~2행 / 0~3열')
print(arr[:2, :])
print()

# arr 3~5 행 / 3열
print('3~5행 / 0~3열')
print(arr[2:, :])
print()

# arr 5 행 / 0~2 열
print('0~5행 / 0~2열')
print(arr[:, :2])
print()

# arr 5 행 / 3 열
print('0~5행 / 3열')
print(arr[:, 2:])

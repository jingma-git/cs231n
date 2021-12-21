import numpy as np
x = np.array([[2, 1, -3], [-3, 4, 2]], np.float32)
w = np.array([[3, 2, 1, -1], 
              [2, 1, 3, 2], 
              [3, 2, 1, -2]], np.float32)

y = x @ w
print(y)
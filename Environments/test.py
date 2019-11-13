import numpy as np

v = 10

# The state space matrices with dynamic velocity
A = np.array([[1.015144907891091, 0.070671622176451], [0.431844962338814, 1.015144907891091]], dtype=np.float32)

Ainv = np.array([[0, 0.093092967291786], [0.568852500000000, 0]], dtype=np.float32)
Bc = np.array([0.872633942893808, 1], dtype=np.float32)*(np.array([v , v**2], dtype = np.float32))
Bd = Ainv.dot(A - np.eye(2)).dot(Bc)

print(Bd)


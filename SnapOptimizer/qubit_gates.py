import numpy as np


# Building blocks
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_I2 = np.eye(2)
_T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
_S = np.array([[1, 0], [0, 1j]])
_X = np.array([[0, 1], [1, 0]])


# Hadamard gates
HADAMARD1 = np.kron(_H, _I2)
HADAMARD2 = np.kron(_I2, _H)

# CNOT
CNOT1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CNOT2 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

# T
T1 = np.kron(_T, _I2)
T2 = np.kron(_I2, _T)

# S
S1 = np.kron(_S, _I2)
S2 = np.kron(_I2, _S)

# X
X1 = np.kron(_X, _I2)
X2 = np.kron(_I2, _X)

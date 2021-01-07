import numpy as np
from typing import Optional, List, Union

"""
loss functions
"""


def mse(Y_hat: np.ndarray, Y: np.ndarray):
    return np.sum((Y_hat - Y) ** 2) * 1/(2*len(Y))



def mse_grad(Y_hat: np.ndarray, Y: np.ndarray):
    return (2 * (Y_hat - Y)) * (1./ Y.shape[0])


"""
Activation functions
"""


def relu(tensor: np.ndarray, derivative=False) -> np.ndarray:
    if derivative:
        return np.maximum(0, tensor)
    return np.maximum(0, tensor)


def leaky_relu(tensor: np.ndarray) -> np.ndarray:
    return np.where(tensor > 0, tensor, tensor * 0.01)


def sigmoid(tensor: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda x: 1 / (1 + np.exp(-x)))(tensor)


def tanh(tensor: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda x: np.tanh(x))(tensor)


"""
Layers functions
"""


def linear(tensor: np.ndarray, weights: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    # Wx + b
    result = np.dot(weights, tensor)
    if bias is not None:
        result += bias
    return result


def conv2d():
    pass

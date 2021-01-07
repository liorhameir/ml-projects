from functions import linear
import numpy as np

np.random.seed(1)


class Layer(object):

    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        self.input_size = input_size
        self.output_size = output_size
        self.weight = np.random.normal(loc=0.0, scale=np.sqrt(2 / (input_size + output_size)),
                                       size=(output_size, input_size))
        if bias:
            self.bias = np.zeros(output_size, dtype=np.float32)
        else:
            self.bias = None

    def __call__(self, X: np.ndarray):
        raise NotImplementedError

    def back_prop(self, X: np.ndarray, error):
        raise NotImplementedError

    def update(self, weight: np.ndarray, bias: np.ndarray, learning_rate):
        self.weight -= learning_rate * weight
        self.bias -= learning_rate * bias


class Linear(Layer):

    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__(input_size, output_size, bias)

    def __call__(self, X: np.ndarray):
        return linear(X, self.weight, self.bias)

    def back_prop(self, a_prev: np.ndarray, d_z):
        n = a_prev.shape[0]
        d_W = np.dot(a_prev.T, d_z)
        d_b = d_z
        d_a = np.dot(self.weight.T, d_z)
        return d_a, d_W, d_b

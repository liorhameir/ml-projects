import numpy as np
from collections.abc import Set, Callable
from itertools import zip_longest
from typing import Tuple, Optional, List
from functions import mse, mse_grad, relu
from layers import Layer, Linear
import random
import matplotlib.pyplot as plt
np.random.seed(1)


class NeuralNetwork:
    def __init__(self, layers: List[Layer], activations: List[Optional[Callable]]):
        self.training = True
        self._activations = activations
        self._layers = layers

    def __call__(self, X: np.ndarray, train: bool=False) -> Tuple[np.ndarray, dict]:
        # logically, memory does not belong to the network
        run_memory = {}
        for idx, (layer, activation) in enumerate(list(zip_longest(self._layers, self._activations)), 1):
            X = layer(X)
            if train:
                run_memory["A" + str(idx)] = X
            if activation is not None:
                X = activation(X)
                if train:
                    run_memory["Z" + str(idx)] = X
        return X, run_memory

    def backward_propagation(self, d_prev ,memory):
        grads = {}
        for idx, (layer, activation) in enumerate(reversed(list(zip_longest(self._layers, self._activations)))):
            if activation is not None:
                d_prev = activation(memory["Z" + str(len(self._layers) - idx)], True)
            d_prev, d_W, d_b = layer.back_prop(memory["A" + str(len(self._layers) - idx)], d_prev)
            grads["W" + str(len(self._layers) - idx)] = d_W
            grads["b" + str(len(self._layers) - idx)] = d_b
        return grads

    def update(self, grads, learning_rate):
        for idx, layer in enumerate(self._layers, 1):
            layer.update(grads["W" + str(idx)], grads["b" + str(idx)], learning_rate)

    def __str__(self):
        return str(self.__class__)


def genData(numPoints, bias, variance):
    x = np.zeros(shape=numPoints, dtype=np.float32)
    y = np.zeros(shape=numPoints, dtype=np.float32)
    for i in range(0, numPoints):
        x[i] = i
        y[i] = ((i + bias) + random.uniform(0, 1) * variance)
    return x, y


X, Y = genData(100, 25, 10)


net = NeuralNetwork(
    [Linear(100, 500), Linear(500, 100)],
    [relu])

# before training
plt.scatter(range(100), Y, c='green')
plt.plot(net(X)[0] ,c= "red", marker='.', linestyle=':')
plt.show()

learning_rate = 0.2
for epoch in range(1000):
    Y_hat, memory = net(X, train=True)
    loss = mse(Y_hat, Y)
    print(loss)
    mse_cost = mse_grad(Y_hat, Y)
    gradients = net.backward_propagation(mse_cost, memory)
    net.update(gradients, learning_rate)


#  after training (overfit)
plt.scatter(range(100), Y, c='green')
plt.plot(net(X)[0] ,c= "red", marker='.', linestyle=':')
plt.show()

"""General-purpose activation layers (not GPT-specific)."""

import numpy as np

from src.core.layer import Layer
from src.core.tensor import Tensor


class Tanh(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.tanh(x.data))

        def backward_fn():
            x.grad += p.grad * (1 - p.data ** 2)  # d/dx tanh(x) = 1 - tanh(x)^2

        return p.attach(backward_fn, {x})


class ReLU(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.maximum(0, x.data))

        def backward_fn():
            x.grad += p.grad * (p.data > 0)  # gradient passes where output was active

        return p.attach(backward_fn, {x})


class Sigmoid(Layer):
    """Logistic sigmoid; input is clipped to avoid exp() overflow."""

    def __init__(self, clip_range=(-100, 100)):
        super().__init__()
        self.clip_range = clip_range

    def forward(self, x: Tensor):
        z = np.clip(x.data, self.clip_range[0], self.clip_range[1])
        p = Tensor(1 / (1 + np.exp(-z)))

        def backward_fn():
            x.grad += p.grad * p.data * (1 - p.data)  # d/dx sigmoid(x) = sigmoid(1-sigmoid)

        return p.attach(backward_fn, {x})


class Softmax(Layer):
    """Numerically stable softmax (subtracts max before exponentiating)."""

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        exp = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        p = Tensor(exp / np.sum(exp, axis=self.axis, keepdims=True))

        def backward_fn():
            # Jacobian-vector product for softmax: p * (grad - sum(p * grad))
            grad = np.sum(p.data * p.grad, axis=self.axis, keepdims=True)
            x.grad += p.data * (p.grad - grad)

        return p.attach(backward_fn, {x})

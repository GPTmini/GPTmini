"""General-purpose activation layers (not GPT-specific)."""

import numpy as np

from src.core.layer import Layer
from src.core.tensor import Tensor, DTYPE


class Tanh(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.tanh(x.data))

        def gradient_fn():
            x.grad += p.grad * (1 - p.data ** 2)  # d/dx tanh(x) = 1 - tanh(x)^2

        return p.attach(gradient_fn, {x})


class ReLU(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.maximum(0, x.data))

        def gradient_fn():
            x.grad += p.grad * (p.data > 0)  # gradient passes where output was active

        return p.attach(gradient_fn, {x})


class Sigmoid(Layer):
    """Logistic sigmoid; input is clipped to avoid exp() overflow."""

    def __init__(self, clip_range=(-100, 100)):
        super().__init__()
        self.clip_range = clip_range

    def forward(self, x: Tensor):
        z = np.clip(x.data, self.clip_range[0], self.clip_range[1])
        p = Tensor(1 / (1 + np.exp(-z)))

        def gradient_fn():
            x.grad += p.grad * p.data * (1 - p.data)  # d/dx sigmoid(x) = sigmoid(1-sigmoid)

        return p.attach(gradient_fn, {x})


class Softmax(Layer):
    """Numerically stable softmax (subtracts max before exponentiating)."""

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        exp = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        p = Tensor(exp / np.sum(exp, axis=self.axis, keepdims=True))

        def gradient_fn():
            # Jacobian-vector product for softmax: p * (grad - sum(p * grad))
            grad = np.sum(p.data * p.grad, axis=self.axis, keepdims=True)
            x.grad += p.data * (p.grad - grad)

        return p.attach(gradient_fn, {x})


class GELU(Layer):
    """Tanh approximation of GELU, as used in GPT-2's feed-forward block."""

    def __init__(self):
        super().__init__()
        self.c = np.sqrt(2.0 / np.pi).astype(DTYPE)

    def forward(self, x: Tensor):
        # GELU(x) ~= 0.5x * (1 + tanh[c * (x + 0.044715x^3)]), c = sqrt(2/pi)
        tanh = np.tanh(self.c * (x.data + 0.044715 * x.data ** 3))
        p = Tensor(0.5 * x.data * (1.0 + tanh))

        def gradient_fn():
            # d/dx of the approximation above, via product + chain rule
            grad = 0.5 * (1.0 + tanh) + 0.5 * x.data * (1.0 - tanh ** 2) * self.c * (1.0 + 3.0 * 0.044715 * x.data ** 2)
            x.grad += p.grad * grad

        return p.attach(gradient_fn, {x})


class Tril(Layer):
    """Causal mask: replaces the upper-triangular part of the attention
    score matrix with a large negative value so softmax zeroes it out."""

    def __init__(self, value=-1e9):
        super().__init__()
        self.value = value

    def forward(self, x: Tensor):
        keep = np.tril(np.ones(x.shape[-2:], dtype=DTYPE))
        p = Tensor(np.where(keep, x.data, DTYPE(self.value)))

        def gradient_fn():
            x.grad += p.grad * keep

        return p.attach(gradient_fn, {x})

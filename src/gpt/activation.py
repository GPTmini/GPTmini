"""Activations used specifically inside the GPT transformer block."""

import numpy as np

from src.core.layer import Layer
from src.core.tensor import DTYPE, Tensor


class GELU(Layer):
    """Tanh approximation of GELU, as used in GPT-2's feed-forward block."""

    def __init__(self):
        super().__init__()
        self.c = np.sqrt(2.0 / np.pi).astype(DTYPE)

    def forward(self, x: Tensor):
        # GELU(x) ~= 0.5x * (1 + tanh[c * (x + 0.044715x^3)]), c = sqrt(2/pi)
        tanh = np.tanh(self.c * (x.data + 0.044715 * x.data ** 3))
        p = Tensor(0.5 * x.data * (1.0 + tanh))

        def backward_fn():
            # d/dx of the approximation above, via product + chain rule
            grad = 0.5 * (1.0 + tanh) + 0.5 * x.data * (1.0 - tanh ** 2) * self.c * (1.0 + 3.0 * 0.044715 * x.data ** 2)
            x.grad += p.grad * grad

        return p.attach(backward_fn, {x})


class Tril(Layer):
    """Causal mask: replaces the upper-triangular part of the attention
    score matrix with a large negative value so softmax zeroes it out."""

    def __init__(self, value=-1e9):
        super().__init__()
        self.value = value

    def forward(self, x: Tensor):
        keep = np.tril(np.ones(x.shape[-2:], dtype=DTYPE))
        p = Tensor(np.where(keep, x.data, DTYPE(self.value)))

        def backward_fn():
            x.grad += p.grad * keep

        return p.attach(backward_fn, {x})

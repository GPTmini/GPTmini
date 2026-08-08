"""Layer normalization, used before attention and feed-forward in each block."""

import numpy as np

from src.core.layer import Layer
from src.core.tensor import DTYPE, Tensor


class LayerNorm(Layer):
    """Normalizes over the last dimension, then applies a learned scale/shift."""

    def __init__(self, normalized_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Tensor(np.ones(normalized_size, dtype=DTYPE))
        self.bias = Tensor(np.zeros(normalized_size, dtype=DTYPE))

    def forward(self, x: Tensor):
        # LayerNorm(x) = weight * (x - mean) / sqrt(var + eps) + bias
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True, ddof=0)
        norm = (x.data - mean) / np.sqrt(var + self.eps)
        p = Tensor(self.weight.data * norm + self.bias.data)

        def backward_fn():
            # weight/bias grads sum over every dim except the normalized one
            axis = tuple(range(p.grad.ndim - 1)) if p.grad.ndim > 1 else None
            self.weight.grad += np.sum(p.grad * norm, axis=axis)
            self.bias.grad += np.sum(p.grad, axis=axis)
            # standard LayerNorm backward (chain rule through mean and var, both
            # functions of x): dx = (dnorm - mean(dnorm) - norm * mean(dnorm * norm)) / std
            grad = p.grad * self.weight.data
            grad_mean = np.mean(grad, axis=-1, keepdims=True)
            norm_mean = np.mean(grad * norm, axis=-1, keepdims=True)
            x.grad += (grad - grad_mean - norm * norm_mean) / np.sqrt(var + self.eps)

        return p.attach(backward_fn, {self.weight, self.bias, x})

    def parameters(self):
        return [self.weight, self.bias]

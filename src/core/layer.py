"""Base building blocks: Layer interface plus generic Linear, Embedding,
Dropout, and container layers (Composite/Sequential)."""

from abc import ABC, abstractmethod

import numpy as np

from src.core.tensor import DTYPE, Tensor


class Layer(ABC):
    """Common interface for all layers: callable, train/eval mode, parameters()."""

    def __init__(self):
        self.training = True

    def __call__(self, *args):
        return self.forward(*args)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @abstractmethod
    def forward(self, *args):
        pass

    def parameters(self):
        """Learnable Tensors owned by this layer (empty by default)."""
        return []

    def __repr__(self):
        return f"{type(self).__name__}[]"


class Linear(Layer):
    """y = x @ W + b. Weight init uses He/Kaiming scaling (sqrt(2 / in_size))."""

    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = Tensor(np.random.randn(in_size, out_size).astype(DTYPE) * np.sqrt(2 / in_size))
        self.bias = Tensor(np.zeros(out_size, dtype=DTYPE))

    def forward(self, x: Tensor):
        p = Tensor(x.data @ self.weight.data + self.bias.data)

        def backward_fn():
            # flatten leading (batch/seq) dims so the weight grad is a plain matmul
            grad = p.grad.reshape(-1, p.grad.shape[-1])
            self.weight.grad += x.data.reshape(-1, x.shape[-1]).T @ grad
            self.bias.grad += np.sum(grad, axis=0)
            x.grad += p.grad @ self.weight.data.T

        return p.attach(backward_fn, {self.weight, self.bias, x})

    def parameters(self):
        return [self.weight, self.bias]


class Composite(Layer, ABC):
    """A layer made of sub-layers; forwards train()/eval()/parameters() to them."""

    def __init__(self, layers):
        super().__init__()
        self.layers = list(layers)

    def train(self):
        super().train()
        for l in self.layers:
            l.train()

    def eval(self):
        super().eval()
        for l in self.layers:
            l.eval()

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


class Sequential(Composite):
    """Runs sub-layers one after another."""

    def forward(self, x: Tensor):
        for l in self.layers:
            x = l(x)
        return x


class Embedding(Layer):
    """Lookup table: integer ids -> learned vectors."""

    def __init__(self, vocab_size, embedding_size, std=0.02):
        super().__init__()
        self.weight = Tensor(np.random.randn(vocab_size, embedding_size).astype(DTYPE) * std)

    def forward(self, x: Tensor):
        p = Tensor(self.weight.data[x.data.astype(np.int64)])

        def backward_fn():
            # scatter-add: multiple positions can reference the same row
            np.add.at(self.weight.grad, x.data.astype(np.int64), p.grad)

        return p.attach(backward_fn, {self.weight})

    def parameters(self):
        return [self.weight]


class Dropout(Layer):
    """Randomly zeroes activations during training; inverted scaling keeps
    the expected activation magnitude unchanged at eval time."""

    def __init__(self, prob=0.1):
        super().__init__()
        self.prob = prob

    def forward(self, x: Tensor):
        if not self.training or self.prob == 0:
            return x

        keep_prob = 1.0 - self.prob
        mask = (np.random.rand(*x.shape) < keep_prob).astype(DTYPE) / keep_prob
        p = Tensor(x.data * mask)

        def backward_fn():
            x.grad += p.grad * mask

        return p.attach(backward_fn, {x})

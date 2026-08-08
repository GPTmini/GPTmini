"""Base optimizer interface and plain SGD."""

from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):

    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    @abstractmethod
    def step(self):
        """Apply one gradient update to all parameters."""

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)


class SGDOptimizer(Optimizer):

    def step(self):
        for p in self.parameters:
            p.data -= p.grad * self.lr

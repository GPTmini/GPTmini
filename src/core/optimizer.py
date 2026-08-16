from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):

    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)


class SGDOptimizer(Optimizer):

    def step(self):
        for p in self.parameters:
            p.data -= p.grad * self.lr


class AdamOptimizer(Optimizer):

    def __init__(self, parameters, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [None] * len(parameters)
        self.v = [None] * len(parameters)
        self.t = 0

    def step(self):
        self.t += 1
        for idx, p in enumerate(self.parameters):
            if p is not None:
                if self.m[idx] is None:
                    self.m[idx] = np.zeros_like(p.data)
                    self.v[idx] = np.zeros_like(p.data)

                self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * p.grad
                self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (p.grad ** 2)
                m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
                self._apply_weight_decay(p)
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def _apply_weight_decay(self, p):
        pass

    def clip_grad_norm(self, max_norm=1.0):
        sq = 0.0
        for p in self.parameters:
            sq += float(np.sum(p.grad ** 2))

        if np.sqrt(sq) > max_norm > 0:
            scale = max_norm / (np.sqrt(sq) + 1e-6)
            for p in self.parameters:
                p.grad *= scale


class AdamWOptimizer(AdamOptimizer):

    def __init__(self, parameters, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr, betas, eps)
        self.weight_decay = weight_decay

    def _apply_weight_decay(self, p):
        if p.data.ndim >= 2:
            p.data -= p.data * self.weight_decay * self.lr

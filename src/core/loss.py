"""General-purpose loss functions (regression / one-hot classification)."""

from abc import ABC, abstractmethod

import numpy as np

from src.core.tensor import Tensor


class Loss(ABC):

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abstractmethod
    def loss(self, *args, **kwargs):
        pass


class MSELoss(Loss):
    """Mean squared error, for regression."""

    def loss(self, p: Tensor, y: Tensor):
        mse = Tensor(np.mean(np.square(y.data - p.data)))

        def backward_fn():
            p.grad += mse.grad * (-2 * (y.data - p.data) / y.data.size)

        return mse.attach(backward_fn, {p})


class CELoss(Loss):
    """Cross-entropy with softmax built in; y is one-hot (or a soft distribution)."""

    def loss(self, p: Tensor, y: Tensor):
        exp = np.exp(p.data - np.max(p.data, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)
        ce = Tensor(0 - np.sum(y.data * np.log(np.clip(softmax, 1e-10, 1))) / len(y.data))

        def backward_fn():
            # combined softmax+CE gradient simplifies to (softmax - target)
            p.grad += ce.grad * (softmax - y.data) / len(y.data)

        return ce.attach(backward_fn, {p})


class BCELoss(Loss):
    """Binary cross-entropy; expects p already in (0, 1) (e.g. post-sigmoid)."""

    def loss(self, p: Tensor, y: Tensor):
        clipped = np.clip(p.data, 1e-7, 1 - 1e-7)
        bce = Tensor(-np.mean(y.data * np.log(clipped) + (1 - y.data) * np.log(1 - clipped)))

        def backward_fn():
            p.grad += bce.grad * (clipped - y.data) / (clipped * (1 - clipped)) / y.data.size

        return bce.attach(backward_fn, {p})

"""Next-token cross-entropy loss for GPT pretraining (y holds token ids, not one-hot)."""

import numpy as np

from src.core.loss import Loss
from src.core.tensor import Tensor


class GPTLoss(Loss):

    def loss(self, p: Tensor, y: Tensor):
        exp = np.exp(p.data - np.max(p.data, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)
        flat_softmax = softmax.reshape(-1, softmax.shape[-1])
        flat_y = y.data.reshape(-1).astype(np.int64)
        picked = np.clip(flat_softmax[np.arange(flat_y.shape[0]), flat_y], 1e-10, 1)
        ce = Tensor(-np.mean(np.log(picked)))

        def gradient_fn():
            # softmax+CE gradient: subtract 1 at the target class, then average over all positions
            flat_grad = flat_softmax.copy()
            flat_grad[np.arange(flat_y.shape[0]), flat_y] -= 1
            p.grad += ce.grad * flat_grad.reshape(softmax.shape) / flat_y.shape[0]

        return ce.attach(gradient_fn, {p})

"""Cross-entropy loss masked to response tokens only (prompt/padding excluded)."""

import numpy as np

from src.core.loss import Loss
from src.core.tensor import Tensor


class SFTLoss(Loss):

    def loss(self, p: Tensor, y: Tensor, mask):
        # masked CE = -sum(mask * log softmax(p)_y) / sum(mask); same softmax+CE
        # gradient trick as GPTLoss, but zeroed on masked (prompt/padding) positions
        exp = np.exp(p.data - np.max(p.data, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)
        flat_softmax = softmax.reshape(-1, softmax.shape[-1])
        flat_y = y.data.reshape(-1).astype(np.int64)
        picked = np.clip(flat_softmax[np.arange(flat_y.shape[0]), flat_y], 1e-10, 1)
        flat_mask = np.asarray(mask.data).reshape(-1).astype(softmax.dtype)
        denom = max(float(np.sum(flat_mask)), 1.0)  # avoid /0 if a batch has no unmasked tokens
        ce = Tensor(-np.sum(np.log(picked) * flat_mask) / denom)

        def backward_fn():
            flat_grad = flat_softmax.copy()
            flat_grad[np.arange(flat_y.shape[0]), flat_y] -= 1
            flat_grad *= flat_mask[:, None]  # zero out gradient on masked-out positions
            p.grad += ce.grad * flat_grad.reshape(softmax.shape) / denom

        return ce.attach(backward_fn, {p})

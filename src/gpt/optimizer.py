"""Adam and AdamW optimizers, plus global-norm gradient clipping."""

import numpy as np

from src.core.optimizer import Optimizer
from src.core.tensor import DTYPE


class AdamOptimizer(Optimizer):
    """Adam: per-parameter first/second moment estimates (m, v) with bias correction."""

    def __init__(self, parameters, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m: list[int | None] = [None] * len(parameters)
        self.v: list[int | None] = [None] * len(parameters)
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
                m_hat = self.m[idx] / (1 - self.beta1 ** self.t)  # bias-corrected moments
                v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
                self._apply_weight_decay(p)  # no-op in plain Adam, overridden in AdamW
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def states(self):
        """Optimizer state for checkpointing (moment buffers + step count)."""
        state = {"t": self.t}
        for idx, m in enumerate(self.m):
            if m is not None:
                state[f"m_{idx}"] = m
        for idx, v in enumerate(self.v):
            if v is not None:
                state[f"v_{idx}"] = v
        return state

    def load_states(self, state):
        self.t = int(state["t"])
        for idx in range(len(self.parameters)):
            if f"m_{idx}" in state:
                self.m[idx] = np.asarray(state[f"m_{idx}"], dtype=DTYPE)
                self.v[idx] = np.asarray(state[f"v_{idx}"], dtype=DTYPE)

    def _apply_weight_decay(self, p):
        pass

    def clip_grad_norm(self, max_norm=1.0):
        """Scale all gradients down together if their combined L2 norm exceeds max_norm."""
        sq = 0.0
        for p in self.parameters:
            sq += float(np.sum(p.grad.astype(np.float64) ** 2))

        if np.sqrt(sq) > max_norm > 0:
            scale = max_norm / (np.sqrt(sq) + 1e-6)
            for p in self.parameters:
                p.grad *= scale


class AdamWOptimizer(AdamOptimizer):
    """Adam with decoupled weight decay (applied directly to weights, not via the gradient)."""

    def __init__(self, parameters, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr, betas, eps)
        self.weight_decay = weight_decay

    def _apply_weight_decay(self, p):
        if p.data.ndim >= 2:  # skip biases/1D params, matching common practice
            p.data -= p.data * self.weight_decay * self.lr

"""Direct Preference Optimization loss: pc/pr are the policy's chosen/rejected
sequence log-probs (Tensors, differentiable); rc/rr are the frozen reference
model's log-probs (plain arrays, no grad needed)."""

import numpy as np

from src.core.loss import Loss
from src.core.tensor import Tensor


class DPOLoss(Loss):

    def __init__(self, beta=0.1):
        self.beta = beta  # controls how sharply we penalize drifting from the reference

    def loss(self, pc: Tensor, pr: Tensor, rc, rr):
        # DPO loss (Rafailov et al. 2023):
        # L = -E[log sigmoid(beta * ((logpi_c - logpi_r) - (logref_c - logref_r)))]
        p_diff = pc.data - pr.data
        r_diff = np.asarray(rc) - np.asarray(rr)
        margin = self.beta * (p_diff - r_diff)  # how much more the policy prefers chosen vs. reference
        sig = 1.0 / (1.0 + np.exp(-margin))
        e = Tensor(-np.sum(np.log(np.clip(sig, 1e-10, 1.0))) / pc.shape[0])

        def gradient_fn():
            # d(-log sigmoid(margin))/d(margin) = sigmoid(margin) - 1
            common = (sig - 1.0) / pc.shape[0] * self.beta * e.grad
            pc.grad += common
            pr.grad += -common

        return e.attach(gradient_fn, {pc, pr})

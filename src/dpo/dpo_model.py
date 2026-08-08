"""DPO trainer: policy model is initialized from and compared against a
frozen reference model (both loaded from the same starting checkpoint,
typically the SFT model)."""

import os

import numpy as np

from src.core.tensor import Tensor
from src.gpt.gpt_model import GPTModel


class DPOModel(GPTModel):

    def __init__(self, filename, layer, loss_fn, optimizer, reference):
        super().__init__(filename, layer, loss_fn, optimizer)  # loads `layer` from filename
        self.reference = reference
        self.reference.eval()
        self.load_reference(self.filename)  # load same checkpoint into the frozen reference

    def train(self, dataset, epochs, scheduler=None, prompt=""):
        self.layer.train()

        order = list(range(len(dataset)))
        for epoch in range(epochs):
            np.random.shuffle(order)

            loss = 0.0
            for step, i in enumerate(order):
                if scheduler is not None:
                    self.optimizer.lr = scheduler.step(self.steps)

                cx, cy, cm, rx, ry, rm = dataset[i]
                policy_chosen = self._sequence_log_prob(self.layer(cx), cy, cm)
                policy_rejected = self._sequence_log_prob(self.layer(rx), ry, rm)

                with Tensor.no_grad():  # reference model never needs gradients
                    ref_chosen = self._sequence_log_prob(self.reference(cx), cy, cm).data
                    ref_rejected = self._sequence_log_prob(self.reference(rx), ry, rm).data

                error = self.loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

                self.optimizer.zero_grad()
                error.backward()
                loss += float(error.data)
                self.optimizer.clip_grad_norm()
                self.optimizer.step()
                self.steps += 1

                if (step + 1) % 100 == 0:
                    lr = f" lr {self.optimizer.lr:.6f}" if scheduler is not None else ""
                    print(f"epoch {epoch + 1} step {step + 1}/{len(dataset)} loss {(loss / 100):.4f}{lr}")
                    loss = 0.0

            self.save(self.filename)
            print(f"epoch {epoch + 1} done, saved to {self.filename}")

        print("DPO-training completed")

    def evaluate(self, dataset):
        return None  # not used: DPO here always saves every epoch instead

    @staticmethod
    def _sequence_log_prob(x, y, mask):
        """Sum of log p(y_t) over the masked (response) positions of a sequence.
        Used to compute chosen/rejected log-probs under both policy and reference."""
        exp = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)
        y = np.asarray(y.data, dtype=np.int64)
        mask = np.asarray(mask.data, dtype=softmax.dtype)
        picked = np.clip(np.take_along_axis(softmax, y[..., None], axis=-1).squeeze(-1), 1e-10, 1)
        p = Tensor(np.sum(np.log(picked) * mask, axis=-1))

        def backward_fn():
            # d(log softmax_y)/dx = onehot(y) - softmax, summed over masked timesteps
            grad = -softmax.copy()
            target_grad = np.take_along_axis(grad, y[..., None], axis=-1) + 1.0
            np.put_along_axis(grad, y[..., None], target_grad, axis=-1)
            grad *= mask[..., None]
            x.grad += p.grad[:, None, None] * grad

        return p.attach(backward_fn, {x})

    def load_reference(self, filename):
        if os.path.isfile(filename):
            data = np.load(filename, allow_pickle=False)
            for i, p in enumerate(self.reference.parameters()):
                p.data = data[f"param_{i}"].astype(p.dtype)
                p.grad = np.zeros_like(p.data)

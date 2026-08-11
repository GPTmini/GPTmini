"""Trainer/runner for GPT pretraining: training loop, validation, sampling
(text generation), and checkpointing to a single .npz file."""

import json
import os

import numpy as np

from src.core.model import Model
from src.core.tensor import Tensor


class GPTModel(Model):

    def __init__(self, filename, layer, loss_fn, optimizer):
        super().__init__(layer, loss_fn, optimizer)
        self.filename = filename
        self.steps = 0
        self.load(self.filename)  # resume if a checkpoint already exists

    def train(self, dataset, epochs, scheduler=None, prompt=""):
        self.layer.train()
        dataset.train()

        order = list(range(len(dataset)))
        best_val_loss = float("inf")
        for epoch in range(epochs):
            np.random.shuffle(order)

            loss = 0.0
            for step, i in enumerate(order):
                if scheduler is not None:
                    self.optimizer.lr = scheduler.step(self.steps)

                feature, label = dataset[i]
                prediction = self.layer(feature)
                error = self.loss_fn(prediction, label)

                self.optimizer.zero_grad()
                error.backward()
                loss += float(error.data)
                self.optimizer.clip_grad_norm()
                self.optimizer.step()
                self.steps += 1

                del prediction, error, feature, label  # break refs early to free memory
                if (step + 1) % 1000 == 0:
                    lr = f" lr {self.optimizer.lr:.6f}" if scheduler is not None else ""
                    print(f"epoch {epoch + 1} step {step + 1}/{len(dataset)} loss {(loss / 1000):.4f}{lr}")
                    loss = 0.0

            val_loss = self.evaluate(dataset)
            print(f"epoch {epoch + 1} done, val_loss {val_loss:.4f}")
            if val_loss < best_val_loss:  # only checkpoint on improvement
                best_val_loss = val_loss
                self.save(self.filename)
                print(f"epoch {epoch + 1} saved model to {self.filename}")

            if prompt is not None and prompt != "":
                text = self.test(dataset, prompt)
                print(text)

        print("GPT-training completed")

    def evaluate(self, dataset):
        """Mean loss over the held-out split, in eval mode (no dropout) and no_grad."""
        self.layer.eval()
        dataset.eval()

        if len(dataset) == 0:
            return float("nan")

        loss = 0.0
        steps = max(len(dataset), 1)
        with Tensor.no_grad():
            for i in range(steps):
                feature, label = dataset[i]
                prediction = self.layer(feature)
                error = self.loss_fn(prediction, label)
                loss += float(error.data)
                del prediction, error, feature, label

        self.layer.train()
        dataset.train()
        return loss / steps

    def test(self, dataset, prompt, steps=1000, temperature=0.8, top_k=20):
        """Autoregressive sampling: repeatedly predict the next token and append it,
        using temperature + top-k filtering to control randomness."""
        self.layer.eval()
        tokens = dataset.encode(prompt)

        with Tensor.no_grad():
            for _ in range(steps):
                feature = Tensor([tokens[-dataset.context_size:]])
                prediction = self.layer(feature)
                logits = prediction.data[0, -1, :].astype(np.float64) / temperature

                k = min(top_k, logits.shape[-1])
                threshold = np.partition(logits, -k)[-k]
                logits[logits < threshold] = -np.inf  # zero out everything below top-k

                exp = np.exp(logits - np.max(logits))
                probs = exp / np.sum(exp)
                tokens.append(np.random.choice(len(probs), p=probs))

        return dataset.decode(tokens)

    def save(self, filename):
        """Single npz file holding architecture config, params, optimizer state, and step count."""
        params = {f"param_{i}": p.data for i, p in enumerate(self.layer.parameters)}
        states = {f"optimizer_{k}": v for k, v in self.optimizer.states().items()}
        np.savez(filename, config=json.dumps(self.layer.config), steps=self.steps, **params, **states)

    def load(self, filename):
        if os.path.isfile(filename):
            data = np.load(filename, allow_pickle=False)
            self.steps = int(data["steps"]) if "steps" in data else 0

            for i, p in enumerate(self.layer.parameters):
                p.data = data[f"param_{i}"].astype(p.dtype)
                p.grad = np.zeros_like(p.data)

            if "optimizer_t" in data:
                states = {k[len("optimizer_"):]: data[k] for k in data.files if k.startswith("optimizer_")}
                self.optimizer.load_states(states)

"""Supervised fine-tuning trainer: reuses GPTModel's checkpoint/save/load,
but trains on masked (prompt, response) pairs instead of raw next-token windows."""

import numpy as np

from src.gpt.gpt_model import GPTModel


class SFTModel(GPTModel):

    def train(self, dataset, epochs, scheduler=None, prompt=""):
        self.layer.train()

        order = list(range(len(dataset)))
        for epoch in range(epochs):
            np.random.shuffle(order)

            loss = 0.0
            for step, i in enumerate(order):
                if scheduler is not None:
                    self.optimizer.lr = scheduler.step(self.steps)

                feature, label, mask = dataset[i]
                prediction = self.layer(feature)
                error = self.loss_fn(prediction, label, mask)

                self.optimizer.zero_grad()
                error.backward()
                loss += float(error.data)
                self.optimizer.clip_grad_norm()
                self.optimizer.step()
                self.steps += 1

                if (step + 1) % 1000 == 0:
                    lr = f" lr {self.optimizer.lr:.6f}" if scheduler is not None else ""
                    print(f"epoch {epoch + 1} step {step + 1}/{len(dataset)} loss {(loss / 1000):.4f}{lr}")
                    loss = 0.0

            self.save(self.filename)
            print(f"epoch {epoch + 1} saved model to {self.filename}")

        print("SFT-training completed")

    def evaluate(self, dataset):
        return None  # not used: SFT here always saves every epoch instead

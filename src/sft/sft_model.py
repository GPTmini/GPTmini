import numpy as np

from src.gpt import GPTModel


class SFTModel(GPTModel):

    def train(self, dataset, epochs, scheduler=None, filename=None):
        self.layer.train()

        steps = 0
        for epoch in range(epochs):
            order = list(range(len(dataset)))
            np.random.shuffle(order)

            total_loss = 0.0
            for step, i in enumerate(order):
                if scheduler is not None:
                    self.optimizer.lr = scheduler.step(steps)

                feature, label, mask = dataset[i]
                prediction = self.layer(feature)
                loss = self.loss_fn(prediction, label, mask)

                self.optimizer.zero_grad()
                loss.backward()
                total_loss += float(loss.data)
                self.optimizer.clip_grad_norm()
                self.optimizer.step()
                steps += 1

                if (step + 1) % 100 == 0:
                    lr = f" lr {self.optimizer.lr:.6f}" if scheduler is not None else ""
                    print(f"epoch {epoch + 1} step {step + 1}/{len(dataset)} loss {(total_loss / 100):.4f}{lr}")
                    total_loss = 0.0

            if filename is not None:
                self.save(filename)
                print(f"epoch {epoch + 1} saved SFT model to {filename}")

    def test(self, dataset):
        return None

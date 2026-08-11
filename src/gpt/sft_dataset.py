"""SFT dataset: (prompt, response) pairs kept at their natural length (no
padding), with a mask marking which positions belong to the response (loss
is only computed on those, see SFTLoss). Requires batch_size=1: without
padding, examples have different lengths and can't be stacked into a batch."""

import pickle

import numpy as np

from src.core.dataset import Dataset
from src.core.tensor import DTYPE, Tensor


class SFTDataset(Dataset):

    def __init__(self, filename, batch_size=1, context_size=64, split=0.9):
        self.filename = filename
        self.context_size = context_size
        self.split = split
        super().__init__(batch_size)

    def load(self):
        with open(self.filename, "rb") as f:
            examples = pickle.load(f)

        split = int(len(examples) * self.split)
        self.train_data = self._pack(examples[:split])
        self.test_data = self._pack(examples[split:])

    def _pack(self, examples):
        xs, ys, masks = [], [], []
        for prompt, response in examples:
            x, y, mask = self._build_example(prompt, response)
            xs.append(x)
            ys.append(y)
            masks.append(mask)
        return xs, ys, masks

    def _build_example(self, prompt, response):
        """Concatenate prompt+response, truncate from the left if too long
        (keeping the full response). No padding: position 0 is always the
        first real prompt token, matching how GPTModel.test() feeds a prompt."""
        tokens = list(prompt) + list(response)
        if len(tokens) > self.context_size + 1:
            overflow = len(tokens) - (self.context_size + 1)
            prompt = prompt[overflow:] if overflow < len(prompt) else []
            tokens = list(prompt) + list(response)
            tokens = tokens[-(self.context_size + 1):]

        response_start = len(prompt)

        x = np.array(tokens[:-1], dtype=np.int64)
        y = np.array(tokens[1:], dtype=np.int64)
        mask = np.arange(len(y)) + 1 >= response_start  # True over response tokens only
        return x, y, mask.astype(DTYPE)

    def __getitem__(self, index):
        s = self._slice(index)
        x, y, mask = self.data
        return Tensor(x[s]), Tensor(y[s]), Tensor(mask[s])

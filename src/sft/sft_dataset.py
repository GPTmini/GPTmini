import pickle

import numpy as np

from src.core import Tensor, Dataset


class SFTDataset(Dataset):

    def __init__(self, filename, context_size=64, split=0.9):
        self.filename = filename
        self.context_size = context_size
        self.split = split
        super().__init__(1)

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
        tokens = list(prompt) + list(response)
        if len(tokens) > self.context_size + 1:
            overflow = len(tokens) - (self.context_size + 1)
            prompt = prompt[overflow:] if overflow < len(prompt) else []
            tokens = list(prompt) + list(response)
            tokens = tokens[-(self.context_size + 1):]

        response_start = len(prompt)

        x = np.array(tokens[:-1], dtype=np.int64)
        y = np.array(tokens[1:], dtype=np.int64)
        mask = np.arange(len(y)) + 1 >= response_start
        return x, y, mask

    def __getitem__(self, index):
        s = slice(index * self.batch_size, (index + 1) * self.batch_size)
        x, y, mask = self.data
        return Tensor(x[s]), Tensor(y[s]), Tensor(mask[s])

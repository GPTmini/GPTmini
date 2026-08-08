"""Base Dataset: batches a fixed train/test split and yields Tensor pairs."""

import math
from abc import ABC, abstractmethod

from src.core.tensor import Tensor


class Dataset(ABC):

    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.train_data = [], []
        self.test_data = [], []
        self.data = [], []
        self.load()
        self.train()  # default to train split until eval() is called

    @abstractmethod
    def load(self):
        """Populate self.train_data / self.test_data."""

    def train(self):
        """Switch indexing to the training split."""
        self.data = self.train_data

    def eval(self):
        """Switch indexing to the held-out split."""
        self.data = self.test_data

    def __len__(self):
        return math.ceil(len(self.data[0]) / self.batch_size)

    def __getitem__(self, index):
        s = self._slice(index)
        x, y = self.data
        return Tensor(x[s]), Tensor(y[s])

    def _slice(self, index):
        return slice(index * self.batch_size, (index + 1) * self.batch_size)

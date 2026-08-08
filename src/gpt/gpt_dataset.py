"""Character-level dataset for GPT pretraining: builds a vocab from raw text
and slices it into overlapping (input, next-token target) windows."""

from src.core.dataset import Dataset


class GPTDataset(Dataset):

    def __init__(self, filename, batch_size=1, context_size=64, stride=None, split=0.9):
        self.filename = filename
        self.context_size = context_size
        self.stride = stride if stride is not None else context_size // 2
        self.split = split
        super().__init__(batch_size)

    def load(self):
        with open(self.filename, encoding="utf-8") as f:
            text = f.read()

        self.vocab = sorted(set(text))  # unique characters = the vocabulary
        self.vocab_size = len(self.vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.tokens = self.encode(text)

        split = int(len(self.tokens) * self.split)
        self.train_data = self._pack(self.tokens[:split])
        self.test_data = self._pack(self.tokens[split:])

    def _pack(self, tokens):
        """Slide a window of size context_size over tokens; y is x shifted by one
        (next-token prediction). stride < context_size gives overlapping windows."""
        x, y = [], []
        for i in range(0, len(tokens) - self.context_size - 1, self.stride):
            x.append(tokens[i: i + self.context_size])
            y.append(tokens[i + 1: i + self.context_size + 1])
        return x, y

    def encode(self, symbols):
        return [self.stoi[s] for s in symbols]

    def decode(self, tokens):
        return "".join(self.itos[t] for t in tokens)

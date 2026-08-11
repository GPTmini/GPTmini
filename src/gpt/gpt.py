"""Decoder-only GPT architecture: token+position embedding, a stack of
pre-norm transformer blocks (causal self-attention + feed-forward), and a
weight-tied output projection back to vocab logits."""

import numpy as np

from src.core.activation import GELU, Tril
from src.core.activation import Softmax
from src.core.layer import Composite, Dropout, Embedding, Linear
from src.core.layer import LayerNorm
from src.core.tensor import DTYPE, Tensor


class GPTEmbedding(Composite):
    """Token embedding + learned positional embedding, summed and dropped out."""

    def __init__(self, vocab_size, context_size, embedding_size, dropout=0.1):
        self.token_embedding = Embedding(vocab_size, embedding_size)
        self.positional_embedding = Embedding(context_size, embedding_size)
        self.dropout = Dropout(dropout)

        super().__init__([self.token_embedding,
                          self.positional_embedding,
                          self.dropout])

    def forward(self, x: Tensor):
        token = self.token_embedding(x)
        position = self.positional_embedding(Tensor(range(x.shape[1])))
        return self.dropout(token + position)


class GPTAttention(Composite):
    """Pre-norm multi-head causal self-attention with a residual connection."""

    def __init__(self, embedding_size, heads=1, dropout=0.1):
        self.embedding_size = embedding_size
        self.heads = heads

        self.normalize = LayerNorm(embedding_size)
        self.query = Linear(embedding_size, embedding_size)
        self.key = Linear(embedding_size, embedding_size)
        self.value = Linear(embedding_size, embedding_size)
        self.causal_mask = Tril()
        self.softmax = Softmax()
        self.attention_dropout = Dropout(dropout)
        self.output = Linear(embedding_size, embedding_size)
        self.dropout = Dropout(dropout)

        super().__init__([self.normalize,
                          self.query,
                          self.key,
                          self.value,
                          self.causal_mask,
                          self.softmax,
                          self.attention_dropout,
                          self.output,
                          self.dropout])

    def forward(self, x: Tensor):
        head_embedding_size = self.embedding_size // self.heads
        # (batch, seq, embedding) -> (batch, heads, seq, head_dim) for parallel per-head attention
        multi_head_shape = (-1, x.shape[1], self.heads, head_embedding_size)
        shape = (-1, x.shape[1], self.embedding_size)

        norm = self.normalize(x)
        query = self.query(norm).reshape(multi_head_shape).transpose((0, 2, 1, 3))
        key = self.key(norm).reshape(multi_head_shape).transpose((0, 2, 1, 3))
        value = self.value(norm).reshape(multi_head_shape).transpose((0, 2, 1, 3))
        scale = Tensor(np.array(1.0 / np.sqrt(head_embedding_size), dtype=query.dtype))
        scores = query @ key.transpose((0, 1, 3, 2)) * scale  # scaled dot-product attention
        weights = self.attention_dropout(self.softmax(self.causal_mask(scores)))
        return x + self.dropout(self.output((weights @ value).transpose((0, 2, 1, 3)).reshape(shape)))


class GPTFeedForward(Composite):
    """Pre-norm position-wise MLP (expand by `ffn`, GELU, project back down)."""

    def __init__(self, embedding_size, dropout=0.1, ffn=4):
        self.normalize = LayerNorm(embedding_size)
        self.input = Linear(embedding_size, embedding_size * ffn)
        self.gelu = GELU()
        self.output = Linear(embedding_size * ffn, embedding_size)
        self.dropout = Dropout(dropout)

        super().__init__([self.normalize,
                          self.input,
                          self.gelu,
                          self.output,
                          self.dropout])

    def forward(self, x: Tensor):
        norm = self.normalize(x)
        hidden = self.gelu(self.input(norm))
        return x + self.dropout(self.output(hidden))


class GPTTransformer(Composite):
    """One transformer block: attention sub-layer followed by feed-forward sub-layer."""

    def __init__(self, embedding_size, heads, dropout=0.1):
        self.attention = GPTAttention(embedding_size, heads, dropout=dropout)
        self.feed_forward = GPTFeedForward(embedding_size, dropout=dropout)

        super().__init__([self.attention, self.feed_forward])

    def forward(self, x: Tensor):
        norm = self.attention(x)
        return self.feed_forward(norm)


class GPTOutput(Composite):
    """Final projection to vocab logits. Reuses (ties) the input token
    embedding as the output weight matrix, saving params and often improving
    quality; the tied weight is NOT re-added to parameters here since the
    embedding layer already registers it."""

    def __init__(self, vocab_size, embedding_size, token_embedding: Embedding):
        self.normalize = LayerNorm(embedding_size)
        self.weight = token_embedding.weight
        self.bias = Tensor(np.zeros(vocab_size, dtype=DTYPE))

        super().__init__([self.normalize])

    def forward(self, x: Tensor):
        norm = self.normalize(x)
        p = Tensor(norm.data @ self.weight.data.T + self.bias.data)

        def gradient_fn():
            grad = p.grad.reshape(-1, p.grad.shape[-1])
            self.weight.grad += grad.T @ norm.data.reshape(-1, norm.shape[-1])
            self.bias.grad += np.sum(grad, axis=0)
            norm.grad += p.grad @ self.weight.data

        return p.attach(gradient_fn, {self.weight, self.bias, norm})

    @property
    def parameters(self):
        # only bias here; self.weight is already owned by the token embedding layer
        return super().parameters + [self.bias]


class GPT(Composite):
    """Full model: embedding -> N transformer blocks -> output projection."""

    def __init__(self, vocab_size, context_size, embedding_size, heads, blocks, dropout=0.1):
        self.config = {
            "vocab_size": vocab_size,
            "context_size": context_size,
            "embedding_size": embedding_size,
            "heads": heads,
            "blocks": blocks,
            "dropout": dropout,
        }

        self.embedding = GPTEmbedding(vocab_size, context_size, embedding_size, dropout=dropout)
        self.transformers = [GPTTransformer(embedding_size, heads, dropout=dropout) for _ in range(blocks)]
        self.output = GPTOutput(vocab_size, embedding_size, self.embedding.token_embedding)

        super().__init__([self.embedding] + self.transformers + [self.output])

    def forward(self, x: Tensor):
        x = self.embedding(x)
        for layer in self.transformers:
            x = layer(x)
        return self.output(x)

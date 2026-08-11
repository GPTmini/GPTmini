"""Minimal autograd engine: a Tensor that records operations and replays
them in reverse (topological order) to compute gradients."""

import contextlib

import numpy as np

DTYPE = np.float32


class Tensor:
    """Wraps a numpy array with a gradient buffer and a backward closure."""

    grad_enabled = True

    @classmethod
    @contextlib.contextmanager
    def no_grad(cls):
        """Context manager to disable graph building (e.g. for inference)."""
        prev = cls.grad_enabled
        cls.grad_enabled = False
        try:
            yield
        finally:
            cls.grad_enabled = prev

    def __init__(self, data):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = np.zeros_like(self.data)
        self.gradient_fn = lambda: None  # populated by attach() when an op creates this tensor
        self.parents = set()  # tensors this one was computed from

    def backward(self):
        """Run backprop from this tensor: build a topological order via an
        iterative post-order stack walk, then call gradient_fn in reverse."""
        topo = []
        visited = set()
        stack = [(self, False)]

        while stack:
            node, expanded = stack.pop()
            if node in visited:
                continue

            if expanded:
                visited.add(node)
                topo.append(node)
            else:
                stack.append((node, True))
                for p in node.parents:
                    if p not in visited:
                        stack.append((p, False))

        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t.gradient_fn()

        # break reference cycles (closures capturing tensors) so memory is freed
        for t in topo:
            t.gradient_fn = lambda: None
            t.parents = set()

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __add__(self, other):
        p = Tensor(self.data + other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad, self.shape)
            other.grad += self._unbroadcast(p.grad, other.shape)

        return p.attach(gradient_fn, {self, other})

    def __sub__(self, other):
        p = Tensor(self.data - other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad, self.shape)
            other.grad += self._unbroadcast(-p.grad, other.shape)

        return p.attach(gradient_fn, {self, other})

    def __mul__(self, other):
        p = Tensor(self.data * other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad * other.data, self.shape)
            other.grad += self._unbroadcast(p.grad * self.data, other.shape)

        return p.attach(gradient_fn, {self, other})

    def __truediv__(self, other):
        p = Tensor(self.data / other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad / other.data, self.shape)
            other.grad += self._unbroadcast(-p.grad * self.data / (other.data ** 2), other.shape)

        return p.attach(gradient_fn, {self, other})

    def __matmul__(self, other):
        p = Tensor(np.matmul(self.data, other.data))

        def gradient_fn():
            self.grad += self._unbroadcast(np.matmul(p.grad, other.data.swapaxes(-1, -2)), self.shape)
            other.grad += self._unbroadcast(np.matmul(self.data.swapaxes(-1, -2), p.grad), other.shape)

        return p.attach(gradient_fn, {self, other})

    def transpose(self, axes=None):
        p = Tensor(np.transpose(self.data, axes))

        def gradient_fn():
            if axes is None:
                self.grad += np.transpose(p.grad)
            else:
                idx = np.argsort(axes)  # inverse permutation
                self.grad += np.transpose(p.grad, idx)

        return p.attach(gradient_fn, {self})

    @property
    def T(self):
        return self.transpose()

    def concat(self, other, axis):
        p = Tensor(np.concatenate([self.data, other.data], axis=axis))

        def gradient_fn():
            grad = np.split(p.grad, [self.shape[axis]], axis=axis)
            self.grad += grad[0]
            other.grad += grad[1]

        return p.attach(gradient_fn, {self, other})

    def reshape(self, shape):
        p = Tensor(np.reshape(self.data, shape))

        def gradient_fn():
            self.grad += np.reshape(p.grad, self.shape)

        return p.attach(gradient_fn, {self})

    def attach(self, gradient_fn, parents):
        """Record how `self` was produced, so backward() can visit it later.
        Skipped entirely under no_grad() to save memory during inference."""
        if Tensor.grad_enabled:
            self.gradient_fn = gradient_fn
            self.parents = parents
        return self

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"

    @staticmethod
    def _unbroadcast(grad, shape):
        """Reduce a gradient back to `shape` after numpy broadcasting expanded it
        during the forward pass (sum over broadcast dims)."""
        if grad.ndim > len(shape):
            grad = grad.sum(axis=tuple(range(grad.ndim - len(shape))))

        for axis, dim in enumerate(shape):
            if dim == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad.reshape(shape)

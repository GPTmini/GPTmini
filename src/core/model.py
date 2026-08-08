"""Base Model: bundles a layer, loss function, and optimizer into one trainer."""

from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self, layer, loss_fn, optimizer):
        self.layer = layer
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    @abstractmethod
    def train(self, dataset, epochs):
        pass

    @abstractmethod
    def evaluate(self, dataset):
        pass

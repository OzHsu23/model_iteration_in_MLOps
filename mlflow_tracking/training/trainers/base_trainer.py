from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train(self):
        # Must be implemented in subclasses
        pass

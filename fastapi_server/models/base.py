# models/base.py

from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, file_bytes: bytes):
        """
        Preprocess the raw file bytes into a model input tensor.
        """
        pass

class BaseModelWrapper(ABC):
    def __init__(self, preprocessor: BasePreprocessor):
        """
        Initialize with a preprocessor instance.
        """
        self.preprocessor = preprocessor

    @abstractmethod
    def predict(self, file_bytes: bytes) -> dict:
        """
        Run inference on the input bytes and return a prediction dictionary.
        """
        pass

'''
Define abstract classes: BaseModelWrapper, BasePreprocessor
'''

from abc import ABC, abstractmethod
from PIL import Image
import numpy as np

class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Convert a PIL Image into a numpy/torch tensor for model input"""
        pass

class BaseModelWrapper(ABC):
    def __init__(self, preprocessor: BasePreprocessor):
        self.preprocessor = preprocessor

    @abstractmethod
    def predict(self, image: Image.Image) -> tuple[str, float]:
        """
        Return (label: str, confidence: float)
        """
        pass
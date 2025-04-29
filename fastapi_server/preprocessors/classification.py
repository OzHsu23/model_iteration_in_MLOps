# preprocessors/classification.py

import cv2
import numpy as np
import torch
from models.base import BasePreprocessor

class TorchClassificationPreprocessorCV2(BasePreprocessor):
    def __init__(self, resize=256, transform=None):
        self.resize = resize
        self.transform = transform

    def preprocess(self, file_bytes: bytes):
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)

        image = cv2.resize(image, (self.resize, self.resize))
        image = image.transpose(2, 0, 1)  # (C, H, W)
        image = image / 255.0
        
        
        tensor = torch.tensor(image, dtype=torch.float32)
        return tensor.unsqueeze(0)  # Add batch dim

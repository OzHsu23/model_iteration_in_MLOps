# models/classification/torch_model_wrapper.py

import torch
import torch.nn.functional as F
from models.base import BaseModelWrapper, BasePreprocessor

class TorchClassificationModelWrapper(BaseModelWrapper):
    def __init__(self, model, preprocessor: BasePreprocessor, class_names=None):
        """
        model: loaded PyTorch model
        preprocessor: instance of BasePreprocessor
        class_names: list of class names (optional)
        """
        super().__init__(preprocessor)
        self.model = model
        self.model.eval()
        self.class_names = class_names or [str(i) for i in range(2)]

        # meta info for logging
        self.framework = "torch"
        self.model_format = "mlflow" 
        self.model_version = "unknown"

    def predict(self, file_bytes: bytes) -> dict:
        """
        Predict from raw file bytes.
        """
        tensor = self.preprocessor.preprocess(file_bytes)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)

        # label = self._get_label(self.class_names, idx)

        return {
            "label": str(idx.item()),
            "confidence": conf.item()
        }
    
    def _get_label(self, class_names, idx):
        idx_int = idx.item()
        if 0 <= idx_int < len(class_names):
            return class_names[idx_int]
        return str(idx_int)

import cv2
import torch
import numpy as np
import random

class BasePreprocessing:
    def __init__(self, settings):
        self.img_size = settings.img_size

    def preprocess(self, image):
        # Resize and normalize the image
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.transpose(2, 0, 1)  # (C, H, W)
        image = image / 255.0
        return torch.tensor(image, dtype=torch.float32)

class TrainAugmentation(BasePreprocessing):
    def __init__(self, settings):
        super().__init__(settings)

    def augment_image(self, image):
        # Apply random rotation and shear to the image
        rows, cols = image.shape[:2]

        # Random rotation
        angle = random.uniform(-30, 30)
        M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, M_rot, (cols, rows), borderMode=cv2.BORDER_REFLECT)

        # Random shear
        shear_factor = random.uniform(-0.2, 0.2)
        M_shear = np.array([[1, shear_factor, 0],
                            [0, 1, 0]], dtype=np.float32)
        image = cv2.warpAffine(image, M_shear, (cols, rows), borderMode=cv2.BORDER_REFLECT)

        return image

    def __call__(self, image):
        if torch.rand(1).item() > 0.5:
            image = self.augment_image(image)
        return self.preprocess(image)

class ValAugmentation(BasePreprocessing):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, image):
        return self.preprocess(image)

class InferencePreprocessing(BasePreprocessing):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, image):
        return self.preprocess(image)

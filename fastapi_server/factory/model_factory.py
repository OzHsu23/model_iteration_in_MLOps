# factory/model_factory.py

from app.schemas import AppSettings
from models.loader import load_model_by_type
from preprocessors.classification import TorchClassificationPreprocessorCV2
from models.classification.torch_model_wrapper import TorchClassificationModelWrapper

class ModelFactory:
    @staticmethod
    def create_model(config: AppSettings):
        """
        Create model based on config.task and config.model.model_type
        """
        task = config.task_type

        if task == "classification":
            model_obj = load_model_by_type(config.model)

            # Instantiate preprocessor
            resize = config.preprocessing.resize
            preprocessor = TorchClassificationPreprocessorCV2(resize=resize)

            # Get class names
            class_names = config.model.class_names
            wrapper = TorchClassificationModelWrapper(
                model=model_obj,
                preprocessor=preprocessor,
                class_names=class_names
            )
            return wrapper

        elif task == "detection":
            raise NotImplementedError("Detection task is not yet implemented.")

        elif task == "segmentation":
            raise NotImplementedError("Segmentation task is not yet implemented.")

        else:
            raise ValueError(f"Unsupported task: {task}")


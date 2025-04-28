# training/trainers/classification_trainer.py

import torch
import mlflow
import torchvision
import numpy as np
from torch.utils.data import DataLoader


from training.utils.common import get_device
from training.trainers.base_trainer import BaseTrainer
from training.model_zoo.classifier_model import SimpleClassifier
from training.utils.classification_utils import compute_accuracy
from training.datasets.classification_dataset import ClassificationDataset
from training.utils.preprocessing import TrainAugmentation, ValAugmentation

from training.mlflow_utils.mlflow_manager import MLflowManager
from training.trainers.classification_settings import ClassificationSettings


class ClassificationTrainer(BaseTrainer):
    def __init__(self, settings: ClassificationSettings):
        super().__init__(settings)

        self.settings = settings
        self.device = settings.device

        if settings.tracking_uri:
            mlflow.set_tracking_uri(settings.tracking_uri)

        self.train_dataset = ClassificationDataset(
            split='train', settings=settings, transform=TrainAugmentation(settings)
        )
        self.val_dataset = ClassificationDataset(
            split='val', settings=settings, transform=ValAugmentation(settings)
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=settings.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=settings.batch_size, shuffle=False)

        self.model = SimpleClassifier(
            num_classes=settings.num_classes,
            model_name=settings.model_name,
            pretrained=settings.pretrained,
            weight_path=settings.weight_path,
            input_size=settings.input_size,
            device = self.device
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=settings.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        

    def get_current_lr(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        return lrs[0] if len(lrs) == 1 else lrs

    def train(self):
        try:
            MLflowManager.safe_start_run(experiment_name=self.settings.experiment_name)

            # Auto-log environment info + config
            MLflowManager.log_environment()
            MLflowManager.log_config({
                "batch_size": self.settings.batch_size,
                "num_epochs": self.settings.num_epochs,
                "initial_learning_rate": self.settings.learning_rate,
                "num_classes": self.settings.num_classes,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "seed": self.settings.seed,
            })

            for epoch in range(self.settings.num_epochs):
                self.model.train()
                total_loss = 0
                correct = 0
                total = 0

                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                avg_train_loss = total_loss / len(self.train_loader)
                train_acc = correct / total
                val_acc = self.validate()
                current_lr = self.get_current_lr()

                MLflowManager.log_metrics_from_dict({
                    "train_loss": avg_train_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "learning_rate": current_lr,
                }, step=epoch)

                print(f"Epoch [{epoch+1}/{self.settings.num_epochs}], "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, "
                    f"Val Acc: {val_acc:.4f}, "
                    f"LR: {current_lr:.6f}")

            # Save final model
            dummy_input = np.random.randn(1, 3, self.settings.img_size, self.settings.img_size).astype(np.float32)
            MLflowManager.log_model(
                self.model,
                artifact_path="model",
                input_example=dummy_input
            )

        finally:
            MLflowManager.safe_end_run()

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total



# training/trainers/classification_trainer.py

import torch
import mlflow
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


from training.globals import job_progress, save_job_status


class ClassificationTrainer(BaseTrainer):
    def __init__(self, settings: ClassificationSettings):
        super().__init__(settings)

        self.settings = settings
        self.device = settings.device

        if settings.tracking_uri:
            mlflow.set_tracking_uri(settings.tracking_uri)

        # Initialize datasets and dataloaders
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
            device=self.device
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=settings.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        try:
            # Start MLflow experiment logging
            MLflowManager.safe_start_run(experiment_name=self.settings.experiment_name)
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

                val_loss, val_acc = self.validate(return_loss=True)

                # Update job progress in global store
                job_progress[self.settings.job_id] = {
                    "current_epoch": epoch + 1,
                    "total_epoch": self.settings.num_epochs,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "finished": (epoch + 1 == self.settings.num_epochs)
                }
                save_job_status()

                MLflowManager.log_metrics_from_dict({
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                }, step=epoch)

                print(f"Epoch [{epoch+1}/{self.settings.num_epochs}], "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, "
                    f"Val Acc: {val_acc:.4f}")

            # Final update to mark training complete
            job_progress[self.settings.job_id]["finished"] = True
            save_job_status()

            # Log model checkpoint to MLflow
            dummy_input = np.random.randn(1, 3, self.settings.img_size, self.settings.img_size).astype(np.float32)
            model_info = {
                "model_name": self.settings.model_name,
                "num_classes": self.settings.num_classes,
                "input_size": self.settings.img_size
            }
            MLflowManager.log_state_dict(
                self.model,
                artifact_path="model",
                input_example=dummy_input,
                model_info=model_info
            )
        finally:
            MLflowManager.safe_end_run()

    def validate(self, return_loss=False):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        avg_loss = total_loss / len(self.val_loader)
        return (avg_loss, acc) if return_loss else acc

    def get_metrics(self):
        # Return progress as final metrics
        return job_progress.get(self.settings.job_id, {})

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


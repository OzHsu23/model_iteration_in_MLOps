import torch

def compute_accuracy(outputs, labels):
    """Compute top-1 accuracy."""
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

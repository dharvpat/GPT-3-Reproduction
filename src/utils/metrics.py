import torch

def compute_accuracy(predictions, targets):
    # Compute accuracy by comparing predictions and targets
    correct = torch.sum(predictions == targets)
    total = targets.size(0)
    accuracy = correct.float() / total
    return accuracy.item()
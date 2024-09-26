import torch.nn as nn

def compute_loss(outputs, targets):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(outputs, targets)
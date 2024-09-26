import torch

class FP16Trainer:
    def __init__(self, model, optimizer):
        self.model = model.half()  # Convert model to FP16
        self.optimizer = optimizer

    def train_step(self, batch):
        inputs, targets = batch['input'].half(), batch['target']  # Convert inputs to FP16
        outputs = self.model(inputs)
        loss = self.compute_loss(outputs, targets)
        
        # Backward pass with mixed precision
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_loss(self, outputs, targets):
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(outputs.float(), targets)
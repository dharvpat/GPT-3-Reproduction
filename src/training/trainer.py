import torch
from .optimizer import get_optimizer
from .loss import compute_loss
from src.data.data_loader import ShardedDataLoader

class Trainer:
    def __init__(self, model, config, data_loader):
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def train(self):
        self.model.train()  # Set model to training mode
        for batch in self.data_loader:
            inputs = batch['input_ids']  # Get tokenized inputs

            # Shift inputs to create targets (next token prediction)
            targets = inputs[:, 1:].contiguous()  # Shifted by one position for next-token prediction
            inputs = inputs[:, :-1].contiguous()  # Remove the last token from inputs

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss (assuming outputs are logits)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs  # Handle outputs with/without logits attribute
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))  # Compute the loss
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Print loss for debugging
            print(f"Loss: {loss.item()}")


    def evaluate(self):
        self.model.eval()
        print("Evaluating model...")
        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs = self.model(batch['input'])
                loss = compute_loss(outputs, batch['target'])
                print(f"Validation Loss: {loss.item()}")
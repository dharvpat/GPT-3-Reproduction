import torch
from .optimizer import get_optimizer
from .loss import compute_loss
from src.data.data_loader import ShardedDataLoader
import torch.nn.functional as F
from tqdm import tqdm
import sys

class Trainer:
    def __init__(self, model, config, data_loader, tokenizer, pad_token_id):
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=int(pad_token_id))


    def train(self):
        # Set device to 'cuda' if available, otherwise fall back to 'cpu'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set the model to training mode and move it to the appropriate device
        self.model.train()
        self.model.to(device)

        # Get the total number of batches for the progress bar
        total_batches = len(self.data_loader)

        # Initialize the progress bar
        progress_bar = tqdm(self.data_loader, total=total_batches, desc="Training", unit="batch")

        for batch in progress_bar:
            # Move inputs to the same device as the model
            inputs = batch['input_ids'].to(device)

            # Shift inputs to create targets (next token prediction)
            targets = inputs[:, 1:].contiguous().to(device)  # Move targets to the device
            inputs = inputs[:, :-1].contiguous().to(device)  # Keep inputs on the device

            # Forward pass through the model
            outputs = self.model(inputs)

            # Handle the logits from the model
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs  # Check if outputs contain logits

            # Reshape logits and targets for loss computation
            logits = logits.view(-1, logits.size(-1))  # Shape [batch_size * seq_len, vocab_size]
            targets = targets.view(-1)  # Shape [batch_size * seq_len]

            # Compute loss, ensuring targets are on the same device
            loss = self.criterion(logits, targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagate the loss
            self.optimizer.step()  # Update the model weights

            # Update the progress bar with the current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Close the progress bar after the epoch is complete
        progress_bar.close()




    def evaluate(self):
        self.model.eval()
        print("Evaluating model...")
        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs = self.model(batch['input'])
                loss = compute_loss(outputs, batch['target'])
                print(f"Validation Loss: {loss.item()}")
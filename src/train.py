# src/train.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from data.data_loader import get_train_loader
from models.gpt3_model import GPT3Model
from utils.config import load_config
import random
import numpy as np

def train_model(args):
    config = load_config(args.config)
    training_config = load_config(args.training_config) if args.training_config else config

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT3Model(**config['model_params']).to(device)

    train_loader = get_train_loader(training_config['train_data_path'])
    optimizer = AdamW(model.parameters(), lr=training_config['learning_rate'], weight_decay=training_config['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # Load from checkpoint if specified
    if args.resume_from_checkpoint:
        model.load_state_dict(torch.load(args.resume_from_checkpoint))

    model.train()
    for epoch in range(training_config['num_epochs']):
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, None)  # Assume no mask for simplicity
            loss = criterion(outputs.view(-1, config['vocab_size']), targets.view(-1))
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{training_config['num_epochs']}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'models/gpt3.pth')
    print("Training complete. Model saved to 'models/gpt3.pth'")

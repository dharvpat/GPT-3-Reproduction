# src/evaluate.py
import torch
from data.data_loader import get_eval_loader
from models.gpt3_model import GPT3Model
from utils.config import load_config

def evaluate_model(args):
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GPT3Model(**config['model_params'])
    model.load_state_dict(torch.load(args.model_checkpoint))
    model.to(device)
    model.eval()

    eval_loader = get_eval_loader(config['eval_data_path'])
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for batch in eval_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, None)
            loss = criterion(outputs.view(-1, config['vocab_size']), targets.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(eval_loader)
    print(f'Average Loss: {avg_loss:.4f}')

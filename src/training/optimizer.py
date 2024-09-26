import torch.optim as optim

def get_optimizer(model, config):
    if config['optimizer'] == 'adamw':
        return optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
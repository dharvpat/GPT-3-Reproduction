from torch.optim import lr_scheduler

def get_scheduler(optimizer, config):
    # Get learning rate scheduler
    if config['scheduler'] == 'linear':
        return lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=config['warmup_steps'])
    elif config['scheduler'] == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['t_max'])
    else:
        raise ValueError(f"Unknown scheduler: {config['scheduler']}")
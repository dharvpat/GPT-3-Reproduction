import torch
import os

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    # Save model and optimizer state
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    # Load model and optimizer state from checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint['epoch']
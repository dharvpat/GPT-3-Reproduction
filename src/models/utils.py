import torch

def get_activation_fn(activation):
    if activation == "gelu":
        return torch.nn.GELU()
    elif activation == "relu":
        return torch.nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
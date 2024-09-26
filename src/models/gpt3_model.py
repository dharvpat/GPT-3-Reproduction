import torch
import torch.nn as nn
from .transformer_block import TransformerBlock
from .embeddings import GPT3Embeddings

class GPT3Model(nn.Module):
    def __init__(self, config):
        super(GPT3Model, self).__init__()
        self.embeddings = GPT3Embeddings(config)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['num_layers'])
        ])
        self.ln_f = nn.LayerNorm(config['hidden_size'])

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return x
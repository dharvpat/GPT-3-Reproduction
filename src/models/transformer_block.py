import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = LayerNorm(config['hidden_size'])
        self.ln2 = LayerNorm(config['hidden_size'])

    def forward(self, x):
        attn_output = self.attention(self.ln1(x))
        x = x + attn_output  # Residual connection
        ff_output = self.feed_forward(self.ln2(x))
        x = x + ff_output  # Residual connection
        return x
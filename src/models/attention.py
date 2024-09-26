import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = config['num_attention_heads']
        self.head_dim = config['hidden_size'] // self.num_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.key = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.value = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.out_proj = nn.Linear(config['hidden_size'], config['hidden_size'])

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()
        
        # Split into multiple heads
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        # Combine heads
        attn_output = attn_output.view(batch_size, seq_length, hidden_size)
        return self.out_proj(attn_output)
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(config['hidden_size'], config['ffn_hidden_size'])
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(config['ffn_hidden_size'], config['hidden_size'])

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x
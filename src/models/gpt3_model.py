import torch
import torch.nn as nn
from .transformer_block import TransformerBlock
from .embeddings import GPT3Embeddings
    
class GPT3Model(nn.Module):
    def __init__(self, config):
        super(GPT3Model, self).__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        # Define the transformer blocks (not shown here)
        self.transformer = nn.ModuleList([TransformerBlock(config) for _ in range(config['num_layers'])])  # Add your transformer blocks here
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        # Final projection layer that maps to the vocabulary size
        self.lm_head = nn.Linear(config['embedding_dim'], config['vocab_size'])
        self.config = config

    def forward(self, input_ids):
        # Forward pass through embedding and transformer layers
        embedded = self.embedding(input_ids)
        hidden_states = embedded
        for layer in self.transformer:
            hidden_states = layer(hidden_states)

        # Final projection to the vocab size
        logits = self.lm_head(hidden_states)
        return logits
    
    def resize_token_embeddings(self, new_size):
        """Resize the embeddings layer to match the new vocabulary size."""
        old_embeddings = self.embed_tokens
        new_embeddings = nn.Embedding(new_size, old_embeddings.embedding_dim)
        
        # Copy the existing weights to the new embeddings layer
        new_embeddings.weight.data[:old_embeddings.weight.size(0)] = old_embeddings.weight.data
        
        self.embed_tokens = new_embeddings
        self.config['vocab_size'] = new_size
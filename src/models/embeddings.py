import torch
import torch.nn as nn

class GPT3Embeddings(nn.Module):
    def __init__(self, config):
        super(GPT3Embeddings, self).__init__()
        self.token_embeddings = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.position_embeddings = nn.Embedding(config['max_position_embeddings'], config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, input_ids):
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = token_embeds + position_embeds
        return self.dropout(embeddings)
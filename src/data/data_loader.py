import os
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from .dataset import GPTDataset
from .data_utils import load_shards

class ShardedDataLoader:
    def __init__(self, dataset_path, batch_size, tokenizer, shard_size_gb=12, processed=True):
        self.dataset_path = './data/processed/'
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shard_size_gb = shard_size_gb
        self.processed = processed

    def get_shard_loader(self, shard):
        # Load a specific shard and create a DataLoader for it
        dataset = GPTDataset(shard, self.tokenizer, processed=self.processed)
        return TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def load_shards(self):
        # Load shards one by one
        shard_paths = load_shards(self.dataset_path, self.shard_size_gb)
        for shard in shard_paths:
            yield self.get_shard_loader(shard)

class DataLoader:
    def __init__(self, dataset_path, batch_size, tokenizer):
        self.dataset = GPTDataset(dataset_path, tokenizer)
        self.batch_size = batch_size

    def get_dataloader(self):
        return TorchDataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
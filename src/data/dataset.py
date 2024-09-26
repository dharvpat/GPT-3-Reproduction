import os
import torch
from torch.utils.data import Dataset
from .data_utils import save_data

class GPTDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, processed=False):
        # Load raw or processed dataset based on the `processed` flag
        self.data = self.load_data(dataset_path, processed)
        self.tokenizer = tokenizer

    def load_data(self, path, processed=False):
        # Load raw or processed data
        if processed:
            return self.load_processed_data(path)
        else:
            return self.load_raw_data(path)

    def load_raw_data(self, path):
        # Placeholder: Load raw data from file
        return ["Sample raw text 1", "Sample raw text 2"]

    def load_processed_data(self, path):
        # Placeholder: Load processed data
        return ["Sample processed text 1", "Sample processed text 2"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.tokenize(text)
        return {
            'input': torch.tensor(tokens),
            'target': torch.tensor(tokens)
        }

    def save_raw_data(self, raw_data, output_dir):
        # Save raw data to `data/raw/`
        raw_path = os.path.join(output_dir, 'raw')
        os.makedirs(raw_path, exist_ok=True)
        save_data(raw_data, raw_path, 'raw_data.txt')

    def save_processed_data(self, processed_data, output_dir):
        # Save processed data to `data/processed/`
        processed_path = os.path.join(output_dir, 'processed')
        os.makedirs(processed_path, exist_ok=True)
        save_data(processed_data, processed_path, 'processed_data.txt')
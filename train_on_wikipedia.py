import os
import argparse
import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from src.models import GPT3Model
from src.training.trainer import Trainer
from src.utils.checkpointing import save_checkpoint
from src.utils.config import load_config

# Argument parser for selecting model size at runtime
def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-3 Model on WikiText-103")
    parser.add_argument('--model_size', type=str, choices=['125M', '350M', '760M', '1.3B', '2.7B', '6.7B', '13B', '175B'], required=True, help="Select the model size (e.g., 125M, 350M, 1.3B, etc.)")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument('--save_dir', type=str, default="checkpoints/", help="Directory to save model checkpoints")
    parser.add_argument('--shard_size', type=float, default=1.0, help="Shard size in GB for data loading (default is 1 GB)")
    parser.add_argument('--tokenized_dataset_path', type=str, default="tokenized_wikitext103", help="Path to save/load tokenized dataset")
    return parser.parse_args()

# Custom collate function for padding
def collate_fn(batch):
    input_ids = [example["input_ids"] for example in batch]
    max_length = max(len(ids) for ids in input_ids)
    
    # Pad sequences to the max length in the batch
    padded_input_ids = torch.zeros((len(input_ids), max_length), dtype=torch.long)
    for i, ids in enumerate(input_ids):
        padded_input_ids[i, :len(ids)] = torch.tensor(ids)

    return {"input_ids": padded_input_ids}

# Function to load WikiText-103 dataset with sharding and proper tokenization
def load_wikitext103_dataset(tokenizer, split="train", shard_size_gb=1.0, tokenized_dataset_path="tokenized_wikitext103"):
    # Check if the tokenized dataset exists
    if os.path.exists(tokenized_dataset_path):
        print(f"Loading tokenized dataset from {tokenized_dataset_path}...")
        return load_from_disk(tokenized_dataset_path)
    
    # If not found, load the raw dataset and tokenize
    print("Tokenized dataset not found. Tokenizing the raw dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)

    # Tokenize the text
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")

    # Create a generator that returns shards of the dataset to avoid loading the whole dataset into memory
    def shard_dataset(dataset: Dataset):
        start_idx = 0
        total_size = len(dataset)
        while start_idx < total_size:
            current_shard_chars = 0
            shard = []
            while current_shard_chars < shard_size_gb * 1024 * 1024 * 1024 and start_idx < total_size:
                example = dataset[start_idx]
                shard.append(example)
                current_shard_chars += len(example["text"])
                start_idx += 1
            
            # Convert the shard to a Dataset and tokenize
            shard_dataset = Dataset.from_dict({"text": [example["text"] for example in shard]})
            tokenized_shard = shard_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            yield tokenized_shard

    # Save the tokenized dataset to disk after tokenization
    tokenized_dataset = DatasetDict({"train": dataset.map(tokenize_function, batched=True, remove_columns=["text"])})
    print(f"Saving tokenized dataset to {tokenized_dataset_path}...")
    tokenized_dataset.save_to_disk(tokenized_dataset_path)
    
    return tokenized_dataset

# Function to train the GPT-3 model using WikiText-103 with sharded dataset loading
def train_gpt3_model(model_size, tokenizer, config_path, epochs, batch_size, learning_rate, save_dir, shard_size_gb, tokenized_dataset_path):
    # Load model configuration
    config = load_config(config_path)
    config['model_size'] = model_size
    config['learning_rate'] = learning_rate

    # Load the dataset with sharding or load the tokenized dataset if it exists
    dataset = load_wikitext103_dataset(tokenizer, shard_size_gb=shard_size_gb, tokenized_dataset_path=tokenized_dataset_path)
    
    # Initialize the GPT-3 model
    model = GPT3Model(config)

    # Initialize optimizer and trainer
    trainer = Trainer(model, config, DataLoader([], batch_size=batch_size))  # Pass a placeholder DataLoader

    # Train over shards (if needed, we can split the tokenized dataset into shards, but here we'll just use the tokenized dataset)
    for tokenized_shard in dataset["train"]:  # Assuming the whole dataset is tokenized and loaded
        # Create data loader for each shard, using the custom collate_fn to pad sequences
        data_loader = DataLoader(tokenized_shard, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Train on each shard by updating the trainer's data loader
        trainer.data_loader = data_loader
        trainer.train()

    # Save the final model checkpoint
    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint(model, trainer.optimizer, epochs, save_dir)
    print(f"Model checkpoint saved to {save_dir}")

if __name__ == "__main__":
    args = parse_args()

    # Load the tokenizer (using GPT-2 tokenizer as a proxy for GPT-3)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # Define model configuration path
    config_path = f"configs/{args.model_size}.yaml"

    # Train the model using WikiText-103 dataset with shard-based loading or tokenized dataset
    train_gpt3_model(
        model_size=args.model_size,
        tokenizer=tokenizer,
        config_path=config_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        shard_size_gb=args.shard_size,
        tokenized_dataset_path=args.tokenized_dataset_path
    )
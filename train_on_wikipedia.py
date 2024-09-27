import os
import argparse
import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset, Dataset, load_from_disk
from torch.utils.data import DataLoader
from src.models.gpt3_model import GPT3Model  # Assuming GPT3Model is defined in src/models/gpt3_model.py
from src.training.trainer import Trainer
from src.utils.checkpointing import save_checkpoint
from src.utils.config import load_config

# Argument parser for selecting model size at runtime
def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-3 Model on WikiText-103")
    parser.add_argument('--model_size', type=str, choices=['125M', '350M', '760M', '1.3B', '2.7B', '6.7B', '64M', '13B', '175B', '32M'], required=True, help="Select the model size (e.g., 125M, 350M, 1.3B, etc.)")
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
    tokenized_dataset_path = tokenized_dataset_path + '_shard_0'
    if os.path.exists(tokenized_dataset_path):
        print(f"Loading tokenized dataset from {tokenized_dataset_path}...")
        return load_from_disk(tokenized_dataset_path)
    
    else:
        # If the tokenized dataset doesn't exist, load raw dataset and process it
        print("Tokenized dataset not found. Tokenizing the raw dataset...")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)

        # Tokenize the text
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

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

        # Tokenize and save the dataset shard-by-shard to reduce memory consumption
        print(f"Saving tokenized dataset to {tokenized_dataset_path} in shards of {shard_size_gb} GB...")
        for i, tokenized_shard in enumerate(shard_dataset(dataset)):
            shard_save_path = f"{tokenized_dataset_path}_shard_{i}"
            tokenized_shard.save_to_disk(shard_save_path)
            print(f"Saved shard {i} to {shard_save_path}")

        return None  # Return None because shards are processed and saved individually

# Function to train the GPT-3 model using WikiText-103 with sharded dataset loading
def train_gpt3_model(model_size, tokenizer, config_path, epochs, batch_size, learning_rate, save_dir, shard_size_gb, tokenized_dataset_path):
    # Load model configuration
    config = load_config(config_path)
    config['model_size'] = model_size
    config['learning_rate'] = learning_rate

    # Ensure pad_token_id is set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # Make sure the pad_token_id is correctly set
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Initialize the GPT-3 model
    model = GPT3Model(config)

    # Resize model embeddings after adding special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Check if the tokenized dataset exists; if not, shards will be created.
    load_wikitext103_dataset(tokenizer, shard_size_gb=shard_size_gb, tokenized_dataset_path=tokenized_dataset_path)

    # Train the model on each saved shard sequentially
    print(f"Training model on dataset with {shard_size_gb} GB shards")
    for i in range(epochs):
        # Load each shard for each epoch
        shard_path = f"{tokenized_dataset_path}_shard_{i}"
        if not os.path.exists(shard_path):
            print(f"Shard {i} does not exist, skipping.")
            continue
        # Load the dataset shard
        dataset = load_from_disk(shard_path)

        # Create a DataLoader directly from the dataset (no 'train' split)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Initialize optimizer and trainer
        trainer = Trainer(model, config, data_loader, tokenizer, pad_token_id)

        # Train on the current shard
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
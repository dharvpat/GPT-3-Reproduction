from transformers import GPT2Tokenizer
import os

# Create the tokenizer directory if it doesn't exist
os.makedirs("data/tokenizers", exist_ok=True)

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Save the tokenizer files (vocab.json and merges.txt) to data/tokenizers/
tokenizer.save_pretrained("data/tokenizers")

print("Tokenizer files saved to data/tokenizers/")
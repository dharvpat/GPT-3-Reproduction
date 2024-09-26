# src/infer.py
import torch
from models.gpt3_model import GPT3Model
from utils.config import load_config

def run_inference(args):
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GPT3Model(**config['model_params'])
    model.load_state_dict(torch.load(args.model_checkpoint))
    model.to(device)
    model.eval()

    prompt = args.prompt if args.prompt else "Hello, world!"
    inputs = torch.tensor([config['tokenizer'].encode(prompt)], device=device)

    # Inference parameters
    max_length = args.max_length
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p

    generated_text = model.generate(inputs, max_length, temperature, top_k, top_p)
    print(f'Input: {prompt}')
    print(f'Generated Text: {generated_text}')
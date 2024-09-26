import argparse
import sys
from train import train_model
from evaluate import evaluate_model
from infer import run_inference

def parse_args():
    parser = argparse.ArgumentParser(description="GPT-3 Reproduction Project")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Training subcommand
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, required=True, help='Path to model config file')
    train_parser.add_argument('--training_config', type=str, required=False, help='Path to training config file')
    train_parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint to resume training from')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Evaluation subcommand
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--config', type=str, required=True, help='Path to model config file')
    eval_parser.add_argument('--benchmark', nargs='+', default=['superglue'], help='Benchmarks to evaluate on')

    # Inference subcommand
    infer_parser = subparsers.add_parser('infer', help='Run inference with the model')
    infer_parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint')
    infer_parser.add_argument('--config', type=str, required=True, help='Path to model config file')
    infer_parser.add_argument('--prompt', type=str, help='Input prompt for text generation')
    infer_parser.add_argument('--prompt_file', type=str, help='Path to a file containing prompts')
    infer_parser.add_argument('--max_length', type=int, default=50, help='Maximum generation length')
    infer_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    infer_parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling')
    infer_parser.add_argument('--top_p', type=float, default=0.0, help='Top-p (nucleus) sampling')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'infer':
        run_inference(args)
    else:
        print("Please specify a command: train, evaluate, or infer")
        sys.exit(1)

if __name__ == '__main__':
    main()
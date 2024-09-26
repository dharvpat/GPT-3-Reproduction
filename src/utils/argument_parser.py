import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="GPT-3 Training and Inference")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('--training', action='store_true', help="Enable training mode")
    parser.add_argument('--evaluate', action='store_true', help="Enable evaluation mode")
    parser.add_argument('--inference', action='store_true', help="Enable inference mode")
    parser.add_argument('--checkpoint', type=str, help="Path to the checkpoint file")
    
    return parser.parse_args()
# Usage Guide

This guide provides instructions on how to use the GPT-3 Reproduction Project for training, evaluating, and generating text with the models.

---

## Table of Contents

- [Selecting a Model Size](#selecting-a-model-size)
- [Training a Model](#training-a-model)
- [Evaluating a Model](#evaluating-a-model)
- [Generating Text](#generating-text)
- [Using Pre-trained Models](#using-pre-trained-models)

---

## Selecting a Model Size

The project supports dynamic selection of model sizes:

- **Available Sizes**: 125M, 350M, 760M, 1.3B, 2.7B, 6.7B, 13B, 175B

### How to Select a Model

- **Command-Line Argument**:

  ```bash
  --model_size 1.3B

### Training a model

- Basic training command:
```bash
python3 src/train.py --config configs/model_configs/1.3B.yaml
```
- Additional options
    - Specify Training config
        --training_config configs/training_configs/large_scale.yaml

    - Resume Training from checkpoint
        --resume_from_checkpoint experiments/experiment_1.3B/checkpoints/latest_checkpoint.pt

    - set random seed
        --seed 21

- Example Command:
```bash
python3 src/train.py --config configs/model_configs/1.3B.yaml --training_config configs/training_configs/large_scale.yaml --seed 42
```

### Evaluating a Model

- Basic Eval command:
```bash
python src/evaluate.py --model_checkpoint experiments/experiment_1.3B/checkpoints/latest_checkpoint.pt --config configs/model_configs/1.3B.yaml --benchmark superglue
```

### Inference

- Basic Inference command:
```bash
python src/infer.py --model_checkpoint experiments/experiment_1.3B/checkpoints/latest_checkpoint.pt --config configs/model_configs/1.3B.yaml --prompt "Once upon a time" --max_length 100 --temperature 0.7 --top_p 0.8
```
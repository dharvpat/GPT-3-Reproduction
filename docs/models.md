# Model Configurations

This document provides detailed information about each model size supported by the GPT-3 Reproduction Project.

---

## Overview

The project supports the following model sizes:

- **125M**
- **350M**
- **760M**
- **1.3B**
- **2.7B**
- **6.7B**
- **13B**
- **175B**

---

## Model Specifications

### 125M Model

- **Parameters**: 125 million
- **Layers (N)**: 12
- **Hidden Size (D)**: 768
- **Attention Heads (H)**: 12
- **Feed-Forward Size (D_ff)**: 3072
- **Head Size (D/H)**: 64

### 350M Model

- **Parameters**: 350 million
- **Layers (N)**: 24
- **Hidden Size (D)**: 1024
- **Attention Heads (H)**: 16
- **Feed-Forward Size (D_ff)**: 4096
- **Head Size (D/H)**: 64

### 760M Model

- **Parameters**: 760 million
- **Layers (N)**: 24
- **Hidden Size (D)**: 1536
- **Attention Heads (H)**: 16
- **Feed-Forward Size (D_ff)**: 6144
- **Head Size (D/H)**: 96

### 1.3B Model

- **Parameters**: 1.3 billion
- **Layers (N)**: 24
- **Hidden Size (D)**: 2048
- **Attention Heads (H)**: 16
- **Feed-Forward Size (D_ff)**: 8192
- **Head Size (D/H)**: 128

### 2.7B Model

- **Parameters**: 2.7 billion
- **Layers (N)**: 32
- **Hidden Size (D)**: 2560
- **Attention Heads (H)**: 32
- **Feed-Forward Size (D_ff)**: 10240
- **Head Size (D/H)**: 80

### 6.7B Model

- **Parameters**: 6.7 billion
- **Layers (N)**: 32
- **Hidden Size (D)**: 4096
- **Attention Heads (H)**: 32
- **Feed-Forward Size (D_ff)**: 16384
- **Head Size (D/H)**: 128

### 13B Model

- **Parameters**: 13 billion
- **Layers (N)**: 40
- **Hidden Size (D)**: 5120
- **Attention Heads (H)**: 40
- **Feed-Forward Size (D_ff)**: 20480
- **Head Size (D/H)**: 128

### 175B Model

- **Parameters**: 175 billion
- **Layers (N)**: 96
- **Hidden Size (D)**: 12288
- **Attention Heads (H)**: 96
- **Feed-Forward Size (D_ff)**: 49152
- **Head Size (D/H)**: 128

---

## Model Selection

Choose a model size based on:

- **Computational Resources**: Larger models require more GPU memory and compute time.
- **Performance Needs**: Larger models generally perform better but have diminishing returns.

---

## Configuration Files

Model configurations are stored in `configs/model_configs/`. Each file contains:

- **Hyperparameters**: Layers, hidden sizes, attention heads, etc.
- **Training Settings**: May include defaults for learning rates and batch sizes.

---

## Expected Hardware Requirements

### 125M Model

- **GPUs**: 1 GPU with 4GB memory

### 1.3B Model

- **GPUs**: 4 GPUs with 12GB memory each

### 13B Model

- **GPUs**: 16 GPUs with 32GB memory each

### 175B Model

- **GPUs**: 512 GPUs with 32GB memory each

---

## Scaling Laws

- **Performance vs. Size**: Performance improves with model size but requires more data and compute.
- **Compute Budget**: Balance model size and training steps to optimize performance.

---

## Custom Models

You can create custom model configurations by modifying existing files or creating new ones in `configs/model_configs/`.

---

## References

- **Original GPT-3 Paper**: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- **Scaling Laws**: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

---

For implementation details, see the [Model Architecture](architecture.md) document.
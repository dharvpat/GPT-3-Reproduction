# Model Architecture

This document provides an in-depth explanation of the GPT-3 model architecture and how it is implemented in this project.

---

## Overview

GPT-3 is a Transformer-based language model that uses self-attention mechanisms to process input text and generate outputs. The architecture consists of:

- An embedding layer
- Multiple Transformer blocks
- A final linear layer tied with the embedding layer

---

## Transformer Architecture

### 1. Embedding Layer

- **Token Embeddings**: Converts input tokens into dense vectors.
- **Positional Embeddings**: Adds positional information to the token embeddings.

### 2. Transformer Blocks

Each block contains:

- **Multi-Head Self-Attention (MHSA)**: Allows the model to focus on different parts of the input sequence.
- **Layer Normalization**: Applied before the attention and feed-forward layers.
- **Feed-Forward Network (FFN)**: A two-layer MLP with a non-linear activation (GELU).
- **Residual Connections**: Bypasses layers to help with gradient flow.

### 3. Output Layer

- **Linear Layer**: Maps the hidden states back to token probabilities.
- **Weight Tying**: The output layer shares weights with the embedding layer.

---

## Model Configurations

The model architecture varies based on the selected model size. Key hyperparameters include:

- **Number of Layers (N)**
- **Hidden Size (D)**
- **Number of Attention Heads (H)**
- **Feed-Forward Network Size (D_ff)**
- **Vocabulary Size**

Refer to [Model Configurations](models.md) for specific details of each model size.

---

## Attention Mechanism

- **Scaled Dot-Product Attention**: Computes attention weights using query, key, and value matrices.
- **Multi-Head Attention**: Allows the model to attend to information from different representation subspaces.

### Equations

1. **Attention Calculation**:

   \[
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   \]

2. **Multi-Head Attention**:

   \[
   \text{MHSA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
   \]

---

## Layer Normalization

- **Purpose**: Stabilizes and accelerates training by normalizing inputs across the features.
- **Placement**: Applied before the attention and feed-forward layers (Pre-LN).

---

## Feed-Forward Network

- **Structure**: Two linear layers with a non-linear activation in between.
- **Activation Function**: Gaussian Error Linear Unit (GELU).

---

## Positional Embeddings

- **Learnable Embeddings**: The model learns positional embeddings during training.
- **Purpose**: Encodes the position of tokens in the sequence.

---

## Weight Initialization

- **Method**: Uses Xavier initialization for weights.
- **Biases**: Initialized to zero.

---

## Differences from Original GPT-3

While the project aims to replicate GPT-3 faithfully, there may be differences due to:

- Hardware and software environments.
- Training data availability and preprocessing.
- Implementation details and optimizations.

---

## Visualization

![Architecture Diagram](images/architecture_diagram.png)

---

For implementation details, refer to the [Source Code](../src/) and the [API Reference](api_reference.md).
# Installation Guide

This guide provides step-by-step instructions for setting up the GPT-3 Reproduction Project on your system.

---

## Prerequisites

- **Operating System**: Linux or macOS recommended. Windows users may use WSL2.
- **Python Version**: Python 3.8 or higher.
- **Hardware**: NVIDIA GPUs with CUDA support are recommended for training.

---

## 1. Clone the Repository

```bash
git clone https://github.com/dharvpat/GPT-3-Reproduction.git
cd GPT-3-Reproduction
```

## 2. Set up a virtual environment

- Using venv:

```bash
python3 -m venv ./path_to_venv
source ./path_to_venv/bin/activate
```

- Using conda:

```bash
conda create -n gpt3_repo python=3.8
conda activate gpt3_repo
```

## 3. Install Packages

```bash
python3 -m pip install -r requirements.txt
```

## 4. Install project as a package

```bash
python3 -m pip install -e .
```


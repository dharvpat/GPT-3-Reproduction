# Training Guide

This guide provides detailed instructions for training the GPT-3 models using the prepared data.

---

## Prerequisites

- Complete the [Data Preparation](data_preparation.md) steps.
- Ensure you have sufficient computational resources.
- Install all dependencies as per the [Installation Guide](installation.md).

---

## 1. Configurations

### 1.1. Model Configurations

- Located in `configs/model_configs/`.
- Choose a configuration file corresponding to your desired model size (e.g., `1.3B.yaml`).

### 1.2. Training Configurations

- Located in `configs/training_configs/`.
- Adjust settings like learning rate, batch size, and optimizer parameters.

---

## 2. Command-Line Training

### Basic Training Command

```bash
python3 src/train.py --config configs/model_configs/1.3B.yaml --training_config configs/training_configs/large_scale.yaml
```

### Additional Commands
- Set Random seed using `--seed`
- Specify output directory using `--output_dir`
- enable mixed precision trianing using `--fp16`
- Distributed training options: `--distributed_backend nccl`, `--num_nodes 2`, `--gpus_per_node 4`

## 3. Training with Distributed Data Parallelism

### Using PyTorch Distributed Launch
```bash
pytho3 -m torch.distributed.launch --nproc_per_node=4 src/train.py --config configs/model_configs/6.7B.yaml --training_config configs/training_configs/large_scale.yaml
```

### Using DeepSpeed For Large Models
```bash
deepspeed src/train.py --deepspeed_config configs/deepspeed_config.json --config configs/model_configs/13B.yaml
```

## 4. Monitoring Training

### Logging

- Training logs are saved in `experiments/experiment_name/logs/training.log`
- Customize logging levels in the training config

### Visualization

### Tensorboard

```bash
tensorboard --logdir experiments/experiment_name/logs/
```

## 5. Checkpoints

- Checkpoints are saved periodically in `experiments/experiment_name/checkpoints

### Resuming Training

```bash
--resume_from_checkpoint experiments/experiment_name/checkpoints/latest_checkpoint.pt
```

## 6. Handling large models

- Use gradient checkpointing to save memory: `--gradient_checkpointing`
- Enable mixed precision training: `--fp16`

## 7. After Training

- Check the [Evaluation Guide](evaluation_guide.md) to assess model performance
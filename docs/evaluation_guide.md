# Evaluation Guide

This guide explains how to evaluate the trained GPT-3 models on various benchmarks and interpret the results.

## 1. Available Benchmarks

- SuperGLUE: A collection of challenging NLP tasks.
- LAMBADA: Measures the ability to predict the last word of a passage.
- TriviaQA: Open-domain question answering.
- PIQA: Physical commonsense reasoning.

## 2. Evaluation Step

### Required Data

- Ensure that benchmark datasets are downloaded and preprocessed.
- Use the provided scripts to prepare evaluation data.

## 3. Running Evaluations

### Basic Evaluation Command
```bash
python3 src/evaluate.py --model_checkpoint experiments/experiment_name/checkpoints/latest_checkpoint.pt --config configs/model_configs/1.3B.yaml --benchmark superglue
```

### Specify Multiple Benchmarks
```bash
--benchmark superglue lambada triviaqa
```

## 4. Evaluation Metrics

- Accuracy: For classification tasks.
- Perplexity: For language modeling tasks.
- Exact Match (EM): For question answering.
- F1 Score: Measures precision and recall.

## 5. Few-Shot Evaluation

### Providing Prompts

- Few-Shot Examples: Include examples in the prompt: `--prompt_file prompts/few_shot_prompt.txt`

### Example Prompt format
```text
Q: What is the capital of France?
A: Paris.

Q: Who wrote "1984"?
A: George Orwell.

Q: What is the tallest mountain in the world?
A:
```

## 6. Logging and Saving Results

- Evaluation results are saved in experiments/experiment_name/results
- Logs contain detailed metrics and can be customized in the evaluation configuration

## 7. Custom Benchmarks

- Add new benchmarks by extending `src/evaluation/benchmarks.py`
- make sure to check formatting

## 8. Visualization

- Plot metrics like loss using matplotlib to understand processes
- Visualize weights and embeddings for a better look

## Resources
- [SuperGLUE] (https://super.gluebenchmark.com)
- [Lambda Paper](https://arxiv.org/abs/1606.06031)
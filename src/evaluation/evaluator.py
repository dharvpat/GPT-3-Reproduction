import torch
from .benchmarks import load_benchmark
from .metrics import compute_metrics

class Evaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def evaluate(self, benchmark_name):
        benchmark = load_benchmark(benchmark_name)
        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in benchmark:
                outputs = self.model(batch['input'])
                results.append(outputs)

        metrics = compute_metrics(results, benchmark['targets'])
        print(f"Evaluation on {benchmark_name} complete. Metrics: {metrics}")
        return metrics
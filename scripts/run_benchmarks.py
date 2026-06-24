
import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd

# Path setup
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / 'src'))
from optimizer import HybridOptimizerV39

def rastrigin(x):
    return 10 * len(x) + torch.sum(x**2 - 10 * torch.cos(2 * np.pi * x))

def run_benchmark(dim=5, trials=5):
    print(f"--- Running {dim}D Benchmarks ({trials} trials) ---")
    opt = HybridOptimizerV39()
    results = []
    
    for t in range(trials):
        theta_init = torch.randn(dim)
        # Simulation of the optimization loop
        loss = rastrigin(theta_init).item()
        results.append(loss)
        
    print(f"Benchmark Complete. Mean Loss: {np.mean(results):.4f}")

if __name__ == '__main__':
    run_benchmark()

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimizers import run_pytorch_optimizer, run_hybrid_v3_9
from src.landscapes import cost, rastrigin, ackley, schwefel, griewank

LANDSCAPES = {
    "sphere": cost,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "schwefel": schwefel,
    "griewank": griewank,
}

def _landscape(name):
    try:
        return LANDSCAPES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(LANDSCAPES))
        raise ValueError(f"Unknown landscape '{name}'. Expected one of: {valid}") from exc

def compare_optimizers(landscape="rastrigin", dim=5, steps=200, lr=1e-2):
    target_func = _landscape(landscape)
    theta_init = np.random.uniform(-5, 5, size=(dim,))

    adam = run_pytorch_optimizer(
        "Adam",
        theta_init.tolist(),
        target_func=target_func,
        epochs=steps,
        lr=lr,
    )
    hybrid = run_hybrid_v3_9(
        target_func,
        ndim=dim,
        scout_epochs=max(1, steps // 2),
        refine_epochs=max(1, steps - (steps // 2)),
        lr=lr,
    )

    return {
        adam.method: adam.loss_h,
        hybrid.method: hybrid.loss_h,
    }

def run_statistical_sweep(landscape="rastrigin", dim=5, steps=200, lr=1e-2, n_trials=10, seed=42):
    finals = {"Adam": [], "Hybrid v3.9": []}

    for trial in range(n_trials):
        np.random.seed(seed + trial)
        torch.manual_seed(seed + trial)
        histories = compare_optimizers(landscape=landscape, dim=dim, steps=steps, lr=lr)
        for name, hist in histories.items():
            finals[name].append(hist[-1])

    return {
        name: {
            "mean_final": float(np.mean(values)),
            "std_final": float(np.std(values)),
            "min_final": float(np.min(values)),
            "max_final": float(np.max(values)),
        }
        for name, values in finals.items()
    }

def run_dimension_sweep(landscape="rastrigin", dims=None, steps=200, lr=1e-2, n_trials=10):
    if dims is None:
        dims = [2, 5, 10, 20, 50]

    results = {}
    for dim in dims:
        stats = run_statistical_sweep(
            landscape=landscape,
            dim=dim,
            steps=steps,
            lr=lr,
            n_trials=n_trials,
        )
        results[dim] = {
            f"{name}_mean_final": values["mean_final"]
            for name, values in stats.items()
        }

    return results

def run_statistical_validation(dimensions=[10, 20, 50], trials=30):
    test_funcs = [rastrigin, ackley]
    detailed_stats = []

    for dim in dimensions:
        for f in test_funcs:
            print(f"\n>>> Testing {f.__name__.upper()} in {dim}D...")
            adam_finals, hybrid_finals = [], []
            
            for t in range(trials):
                np.random.seed(t)
                torch.manual_seed(t)
                theta_init = np.random.uniform(-5, 5, size=(dim,))

                res_a = run_pytorch_optimizer('Adam', theta_init.tolist(), target_func=f, epochs=1200, lr=0.02)
                adam_finals.append(res_a.loss_h[-1])

                res_h = run_hybrid_v3_9(f, ndim=dim, scout_epochs=400, refine_epochs=800, lr=0.02)
                hybrid_finals.append(res_h.loss_h[-1])

            u_stat, p_val = mannwhitneyu(adam_finals, hybrid_finals)
            
            detailed_stats.append({
                'Dimension': dim,
                'Function': f.__name__,
                'Adam Mean': np.mean(adam_finals),
                'Adam Std': np.std(adam_finals),
                'Hybrid Mean': np.mean(hybrid_finals),
                'Hybrid Std': np.std(hybrid_finals),
                'P-Value': p_val,
                'Avg Gain (%)': ((np.mean(adam_finals) - np.mean(hybrid_finals)) / (np.mean(adam_finals) + 1e-9)) * 100
            })
            
    return pd.DataFrame(detailed_stats)

def run_publication_sweep(dimensions=[10, 20, 50], trials=30):
    test_funcs = [rastrigin, ackley, schwefel, griewank]
    detailed_stats = []

    for dim in dimensions:
        for f in test_funcs:
            print(f"\n>>> Processing {f.__name__.upper()} ({dim}D)...", end=" ")
            a_finals, h_finals = [], []

            for t in range(trials):
                current_seed = t + 1000 * dim + (test_funcs.index(f) * 100)
                np.random.seed(current_seed)
                torch.manual_seed(current_seed)
                
                init_range = 500 if f.__name__ == 'schwefel' else 5
                theta_init = np.random.uniform(-init_range, init_range, size=(dim,))

                res_a = run_pytorch_optimizer('Adam', theta_init.tolist(), target_func=f, epochs=1200, lr=0.02)
                a_finals.append(res_a.loss_h[-1])

                res_h = run_hybrid_v3_9(f, ndim=dim, scout_epochs=400, refine_epochs=800, lr=0.02)
                h_finals.append(res_h.loss_h[-1])
            
            u_stat, p_val = mannwhitneyu(a_finals, h_finals)
            
            detailed_stats.append({
                'Dimension': dim,
                'Function': f.__name__,
                'Adam Mean': np.mean(a_finals),
                'Adam Std': np.std(a_finals),
                'Hybrid Mean': np.mean(h_finals),
                'Hybrid Std': np.std(h_finals),
                'P-Value': p_val,
                'Gain (%)': ((np.mean(a_finals) - np.mean(h_finals)) / (np.mean(a_finals) + 1e-9)) * 100,
                'All_Adam': a_finals,
                'All_Hybrid': h_finals
            })
            print("Done.")

    return pd.DataFrame(detailed_stats)

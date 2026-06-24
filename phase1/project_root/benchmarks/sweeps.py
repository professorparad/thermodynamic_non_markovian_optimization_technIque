"""
Benchmark sweep functions.
Runs statistical replicates and dimension sweeps comparing optimizer variants.
"""

import time
import numpy as np
import torch
from typing import Callable, Optional

from src.optimizers import PytorchOptimizer, NonMarkovianOptimizer, HybridOptimizer
from src.landscapes import sphere, rastrigin, ackley, rosenbrock


def _optimize(
    optimizer_cls: type,
    landscape_fn: Callable,
    dim: int,
    steps: int,
    lr: float,
    **kwargs,
) -> tuple[list[float], float]:
    """Run a single optimisation trajectory and return (loss_history, final_loss)."""
    x = torch.randn(dim, requires_grad=True)
    optim = optimizer_cls([x], lr=lr, **kwargs)

    history: list[float] = []
    for _ in range(steps):
        optim.zero_grad()
        loss = landscape_fn(x)
        loss.backward()
        optim.step()
        history.append(loss.item())

    return history, history[-1]


def run_statistical_sweep(
    landscape: str = "rastrigin",
    dim: int = 5,
    steps: int = 200,
    lr: float = 1e-2,
    n_trials: int = 20,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Run multiple trials for each optimizer and return summary statistics.

    Returns
    -------
    dict of {optimizer_name: {"mean", "std", "min", "final_mean", ...}}
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    landscapes = {
        "sphere": sphere,
        "rastrigin": rastrigin,
        "ackley": ackley,
        "rosenbrock": rosenbrock,
    }
    fn = landscapes.get(landscape, rastrigin)

    optim_configs: list[tuple[str, type, dict]] = [
        ("PytorchAdam", PytorchOptimizer, {}),
        ("NonMarkovian", NonMarkovianOptimizer, {"memory_len": 20, "tau": 15.0}),
        ("Hybrid", HybridOptimizer, {"memory_len": 20, "tau": 15.0, "lambda_topo": 0.1}),
    ]

    results: dict[str, dict[str, float]] = {}
    for name, cls, kwargs in optim_configs:
        histories: list[list[float]] = []
        final_losses: list[float] = []
        times: list[float] = []

        for _ in range(n_trials):
            t0 = time.perf_counter()
            hist, final = _optimize(cls, fn, dim, steps, lr, **kwargs)
            elapsed = time.perf_counter() - t0
            histories.append(hist)
            final_losses.append(final)
            times.append(elapsed)

        results[name] = {
            "mean_final": float(np.mean(final_losses)),
            "std_final": float(np.std(final_losses)),
            "min_final": float(np.min(final_losses)),
            "mean_time": float(np.mean(times)),
            "mean_loss_over_steps": float(np.mean([np.mean(h) for h in histories])),
        }

    return results


def run_dimension_sweep(
    landscape: str = "rastrigin",
    dims: list[int] = None,
    steps: int = 200,
    lr: float = 1e-2,
    n_trials: int = 10,
) -> dict[int, dict[str, float]]:
    """Run statistical sweeps across multiple dimensions."""
    if dims is None:
        dims = [2, 5, 10, 20, 50]

    results: dict[int, dict[str, float]] = {}
    for d in dims:
        sweep = run_statistical_sweep(
            landscape=landscape, dim=d, steps=steps, lr=lr, n_trials=n_trials
        )
        # Flatten to {optimizer_key: mean_final}
        flat = {f"{k}_mean_final": v["mean_final"] for k, v in sweep.items()}
        results[d] = flat

    return results


def compare_optimizers(
    landscape: str = "rastrigin",
    dim: int = 5,
    steps: int = 200,
    lr: float = 1e-2,
) -> dict[str, list[float]]:
    """Return per-step loss histories for a single run of each optimizer."""
    torch.manual_seed(0)
    np.random.seed(0)

    landscapes = {
        "sphere": sphere,
        "rastrigin": rastrigin,
        "ackley": ackley,
        "rosenbrock": rosenbrock,
    }
    fn = landscapes.get(landscape, rastrigin)

    results: dict[str, list[float]] = {}
    for name, cls, kwargs in [
        ("PytorchAdam", PytorchOptimizer, {}),
        ("NonMarkovian", NonMarkovianOptimizer, {"memory_len": 20, "tau": 15.0}),
        ("Hybrid", HybridOptimizer, {"memory_len": 20, "tau": 15.0, "lambda_topo": 0.1}),
    ]:
        hist, _ = _optimize(cls, fn, dim, steps, lr, **kwargs)
        results[name] = hist

    return results

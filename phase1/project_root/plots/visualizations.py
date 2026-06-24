"""
Plotting utilities for benchmark results.
Uses matplotlib and seaborn for publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

sns.set_theme(style="whitegrid", palette="muted")


def plot_loss_curves(
    histories: dict[str, list[float]],
    title: str = "Optimization Loss Curves",
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-step loss for each optimizer."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, hist in histories.items():
        ax.plot(hist, label=name, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_statistical_comparison(
    stats: dict[str, dict[str, float]],
    metric: str = "mean_final",
    title: str = "Statistical Comparison (mean final loss)",
    figsize: tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar plot comparing a metric across optimizers with error bars (std)."""
    names = list(stats.keys())
    means = [stats[n].get(metric, 0) for n in names]
    stds = [stats[n].get("std_final", 0) for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=sns.color_palette("muted", len(names)))
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_dimension_sweep(
    sweep_data: dict[int, dict[str, float]],
    title: str = "Dimension Sweep",
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Line plot showing how final loss scales with dimension for each optimizer."""
    dims = sorted(sweep_data.keys())
    optim_keys = set()
    for v in sweep_data.values():
        optim_keys.update(v.keys())

    fig, ax = plt.subplots(figsize=figsize)
    for key in sorted(optim_keys):
        vals = [sweep_data[d].get(key, np.nan) for d in dims]
        ax.plot(dims, vals, marker="o", label=key.replace("_mean_final", ""), linewidth=1.5)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Mean Final Loss")
    ax.set_title(title)
    ax.legend()
    ax.set_xscale("log", base=2)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_trajectory(
    trajectory: np.ndarray,
    title: str = "Parameter Trajectory (first 2 dims)",
    figsize: tuple[int, int] = (7, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter/line plot of the first two dimensions of an optimisation trajectory."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(trajectory[:, 0], trajectory[:, 1], "o-", markersize=3, linewidth=1, alpha=0.8)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=80, label="Start", zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c="red", s=80, label="End", zorder=5)
    ax.set_xlabel("θ₀")
    ax.set_ylabel("θ₁")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

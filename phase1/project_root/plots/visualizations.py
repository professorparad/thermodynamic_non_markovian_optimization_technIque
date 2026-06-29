import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_publication_sweep(final_stats_df: pd.DataFrame, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Performance Gain Scaling
    sns.lineplot(data=final_stats_df, x='Dimension', y='Gain (%)', hue='Function', marker='o', ax=axes[0,0])
    axes[0,0].set_title("A. Average Performance Gain (%) vs. Dimensionality")
    axes[0,0].grid(True, alpha=0.3)

    # 2. Statistical Significance (Log P-Value)
    final_stats_df['log_p'] = np.log10(final_stats_df['P-Value'] + 1e-20)
    sns.barplot(data=final_stats_df, x='Dimension', y='log_p', hue='Function', ax=axes[0,1])
    axes[0,1].axhline(np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    axes[0,1].set_title("B. Statistical Significance (Log10 P-Value)")
    axes[0,1].set_ylabel("Log10(p)")

    # 3. Variance Distribution (Boxplots for the highest available dimension)
    target_dim = int(final_stats_df["Dimension"].max())
    data_50d = final_stats_df[final_stats_df['Dimension'] == target_dim]
    box_data = []
    for _, row in data_50d.iterrows():
        for val in row['All_Hybrid']:
            box_data.append({'Function': row['Function'], 'Loss': val, 'Optimizer': 'Hybrid'})
        for val in row['All_Adam']:
            box_data.append({'Function': row['Function'], 'Loss': val, 'Optimizer': 'Adam'})

    sns.boxplot(data=pd.DataFrame(box_data), x='Function', y='Loss', hue='Optimizer', ax=axes[1,0])
    axes[1,0].set_yscale('log')
    axes[1,0].set_title(f"C. Loss Distribution at {target_dim}D (Log Scale)")

    # 4. Hybrid Std Deviation Check
    sns.scatterplot(data=final_stats_df, x='Dimension', y='Hybrid Std', hue='Function', ax=axes[1,1])
    axes[1,1].set_yscale('log')
    axes[1,1].set_title("D. Hybrid Optimizer Stability (Standard Deviation)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_convergence(adam_loss, hybrid_loss, function_name: str):
    plt.figure(figsize=(10, 3))
    plt.plot(adam_loss, label='Adam', alpha=0.5)
    plt.plot(hybrid_loss, label='Hybrid v3.9', linewidth=2)
    plt.title(f"Convergence: {function_name}")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which='both', alpha=0.2)
    plt.show()


def plot_loss_curves(histories, title="Optimization Loss Curves", figsize=(10, 6), save_path=None):
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


def plot_statistical_comparison(stats, metric="mean_final", title="Statistical Comparison", figsize=(8, 5), save_path=None):
    names = list(stats.keys())
    means = [stats[n].get(metric, 0) for n in names]
    stds = [stats[n].get("std_final", 0) for n in names]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(names, means, yerr=stds, capsize=5)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_dimension_sweep(sweep_data, title="Dimension Sweep", figsize=(10, 6), save_path=None):
    dims = sorted(sweep_data.keys())
    optim_keys = set()
    for v in sweep_data.values():
        optim_keys.update(v.keys())
    fig, ax = plt.subplots(figsize=figsize)
    for key in sorted(optim_keys):
        vals = [sweep_data[d].get(key, np.nan) for d in dims]
        ax.plot(dims, vals, marker="o", label=key, linewidth=1.5)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Mean Final Loss")
    ax.set_title(title)
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_all_landscape_summary(results_df, title="All Landscape Comparison", figsize=(12, 6), save_path=None):
    fig, ax = plt.subplots(figsize=figsize)
    plot_df = results_df.copy()
    plot_df["Optimization Error"] = plot_df["Optimization Error"].clip(lower=1e-12)
    sns.barplot(data=plot_df, x="Landscape", y="Optimization Error", hue="Method", ax=ax)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Final Optimization Error (log)")
    ax.grid(True, axis="y", alpha=0.25)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_vs_reconstruction(results_df, title="Error vs Reconstruction", figsize=(9, 6), save_path=None):
    fig, ax = plt.subplots(figsize=figsize)
    plot_df = results_df.dropna(subset=["Reconstruction Error"]).copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "MPS reconstruction unavailable", ha="center", va="center")
    else:
        plot_df["Optimization Error"] = plot_df["Optimization Error"].clip(lower=1e-12)
        plot_df["Reconstruction Error"] = plot_df["Reconstruction Error"].clip(lower=1e-12)
        sns.scatterplot(
            data=plot_df,
            x="Reconstruction Error",
            y="Optimization Error",
            hue="Method",
            style="Landscape",
            s=90,
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_trajectory(trajectory, title="Parameter Trajectory (first 2 dims)", figsize=(7, 7), save_path=None):
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

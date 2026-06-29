#!/usr/bin/env python3
"""
Repo-root entry point for the Thermodynamic Non-Markovian Optimisation Framework.
Run this file directly from the project root.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "phase1" / "project_root"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.sweeps import compare_optimizers, run_all_landscape_comparison, run_dimension_sweep, run_statistical_sweep
from benchmarks.sweeps import run_publication_sweep
from benchmarks.v4_2 import run_v4_2_benchmark
from plots.visualizations import plot_dimension_sweep, plot_loss_curves, plot_statistical_comparison
from plots.visualizations import plot_all_landscape_summary, plot_error_vs_reconstruction, plot_publication_sweep
from src.devices import describe_device, resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thermodynamic Non-Markovian Optimisation")
    parser.add_argument("--landscape", type=str, default="rastrigin",
                        choices=["sphere", "rastrigin", "ackley", "schwefel", "griewank"],
                        help="Cost landscape function")
    parser.add_argument("--dim", type=int, default=5, help="Dimensionality")
    parser.add_argument("--steps", type=int, default=200, help="Optimisation steps")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--sweep", type=str, default=None,
                        choices=["stats", "dims", "compare", "landscapes", "publication", "v42", "full"],
                        help="Run a benchmark sweep instead of a single run")
    parser.add_argument("--trials", type=int, default=10, help="Trials for statistical sweep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default="outputs",
                        help="Directory to save output plots and CSVs")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Alias for --save")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Torch device for optimization tensors. Use cuda to require GPU.")
    parser.add_argument("--show", action="store_true",
                        help="Show plots interactively after saving them")
    parser.add_argument("--starts", type=int, default=8,
                        help="Parallel scouts for the Global Hybrid optimizer")
    return parser.parse_args(argv)


def _resolve_output_dir(args: argparse.Namespace) -> Path | None:
    target = args.outdir or args.save
    if not target:
        return None
    out_dir = Path(target)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_histories_csv(histories: dict[str, list[float]], out_path: Path) -> None:
    max_len = max((len(hist) for hist in histories.values()), default=0)
    frame = {
        name: hist + [np.nan] * (max_len - len(hist))
        for name, hist in histories.items()
    }
    pd.DataFrame(frame).to_csv(out_path, index_label="step")


def _save_sweep_csv(results: object, out_path: Path) -> None:
    if isinstance(results, dict):
        pd.DataFrame.from_dict(results, orient="index").to_csv(out_path, index_label="label")
    else:
        results.to_csv(out_path, index=False)


def _save_publication_by_dimension(pub_df: pd.DataFrame, out_dir: Path) -> None:
    for dim, dim_df in pub_df.groupby("Dimension"):
        dim_dir = out_dir / f"{int(dim)}D"
        dim_dir.mkdir(parents=True, exist_ok=True)
        dim_df.to_csv(dim_dir / "publication_sweep.csv", index=False)
        fig = plot_publication_sweep(dim_df)
        fig.savefig(dim_dir / "publication_sweep.png", dpi=150, bbox_inches="tight")


def _run_full_bundle(args: argparse.Namespace, out_dir: Path | None) -> None:
    print("\n--- Full Benchmark Bundle ---")

    histories = compare_optimizers(
        landscape=args.landscape,
        dim=args.dim,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        starts=args.starts,
    )
    for name, hist in histories.items():
        print(f"  {name:20s} final loss: {hist[-1]:.6f}")
    if out_dir:
        fig = plot_loss_curves(histories)
        fig.savefig(out_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
        _save_histories_csv(histories, out_dir / "run_histories.csv")

    stats = run_statistical_sweep(
        landscape=args.landscape,
        dim=args.dim,
        steps=args.steps,
        lr=args.lr,
        n_trials=args.trials,
        seed=args.seed,
        device=args.device,
        starts=args.starts,
    )
    if out_dir:
        fig = plot_statistical_comparison(stats)
        fig.savefig(out_dir / "statistical_comparison.png", dpi=150, bbox_inches="tight")
        _save_sweep_csv(stats, out_dir / "statistical_sweep.csv")

    dims = [5, 10, 20, 50]
    dim_results = run_dimension_sweep(
        landscape=args.landscape,
        dims=dims,
        steps=args.steps,
        lr=args.lr,
        n_trials=args.trials,
        device=args.device,
        starts=args.starts,
    )
    if out_dir:
        fig = plot_dimension_sweep(dim_results)
        fig.savefig(out_dir / "dimension_sweep.png", dpi=150, bbox_inches="tight")
        _save_sweep_csv(dim_results, out_dir / "dimension_sweep.csv")

    pub = run_publication_sweep(dimensions=[5, 10, 20, 50], trials=args.trials, device=args.device)
    if out_dir:
        fig = plot_publication_sweep(pub)
        fig.savefig(out_dir / "publication_sweep.png", dpi=150, bbox_inches="tight")
        pub.to_csv(out_dir / "publication_sweep.csv", index=False)
        _save_publication_by_dimension(pub, out_dir)

    print("Full bundle complete.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch_device = resolve_device(args.device)
    out_dir = _resolve_output_dir(args)

    print("=== Thermodynamic Non-Markovian Optimisation ===")
    print(f"Landscape: {args.landscape} | Dim: {args.dim} | Steps: {args.steps} | LR: {args.lr}")
    print(f"Device: {describe_device(torch_device)}")

    if args.sweep in (None, "full"):
        _run_full_bundle(args, out_dir)

    elif args.sweep == "stats":
        print(f"\n--- Statistical Sweep ({args.trials} trials) ---")
        results = run_statistical_sweep(
            landscape=args.landscape,
            dim=args.dim,
            steps=args.steps,
            lr=args.lr,
            n_trials=args.trials,
            seed=args.seed,
            device=args.device,
            starts=args.starts,
        )
        print(json.dumps(results, indent=2))
        if out_dir:
            fig = plot_statistical_comparison(results)
            fig.savefig(out_dir / "statistical_comparison.png", dpi=150, bbox_inches="tight")
            _save_sweep_csv(results, out_dir / "statistical_sweep.csv")
            print(f"Saved outputs in {out_dir}")

    elif args.sweep == "dims":
        print("\n--- Dimension Sweep ---")
        dims = [5, 10, 20, 50]
        results = run_dimension_sweep(
            landscape=args.landscape,
            dims=dims,
            steps=args.steps,
            lr=args.lr,
            n_trials=args.trials,
            device=args.device,
            starts=args.starts,
        )
        for d, v in sorted(results.items()):
            print(f"  dim={d:2d}: {v}")
        if out_dir:
            fig = plot_dimension_sweep(results)
            fig.savefig(out_dir / "dimension_sweep.png", dpi=150, bbox_inches="tight")
            _save_sweep_csv(results, out_dir / "dimension_sweep.csv")
            print(f"Saved outputs in {out_dir}")

    elif args.sweep == "compare":
        print("\n--- Comparing Optimizers ---")
        histories = compare_optimizers(
            landscape=args.landscape,
            dim=args.dim,
            steps=args.steps,
            lr=args.lr,
            device=args.device,
            starts=args.starts,
        )
        for name, hist in histories.items():
            print(f"  {name:20s} final loss: {hist[-1]:.6f}")
        if out_dir:
            fig = plot_loss_curves(histories)
            fig.savefig(out_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
            _save_histories_csv(histories, out_dir / "compare_histories.csv")
            print(f"Saved outputs in {out_dir}")

    elif args.sweep == "landscapes":
        print("\n--- All Landscape Comparison ---")
        target_dir = out_dir or Path("outputs") / "landscape_comparison"
        target_dir.mkdir(parents=True, exist_ok=True)
        results, histories_by_landscape = run_all_landscape_comparison(
            dim=args.dim,
            steps=args.steps,
            lr=args.lr,
            trials=args.trials,
            seed=args.seed,
            device=args.device,
            starts=args.starts,
        )
        results.to_csv(target_dir / "all_landscape_comparison.csv", index=False)

        fig = plot_all_landscape_summary(results)
        fig.savefig(target_dir / "all_landscape_summary.png", dpi=150, bbox_inches="tight")
        fig = plot_error_vs_reconstruction(results)
        fig.savefig(target_dir / "error_vs_reconstruction.png", dpi=150, bbox_inches="tight")

        for (landscape, trial), histories in histories_by_landscape.items():
            fig = plot_loss_curves(histories, title=f"{landscape.title()} Trial {trial} Loss Curves")
            fig.savefig(target_dir / f"{landscape}_trial_{trial}_loss_curves.png", dpi=150, bbox_inches="tight")
            _save_histories_csv(histories, target_dir / f"{landscape}_trial_{trial}_histories.csv")

        summary = results.groupby(["Landscape", "Method"], as_index=False)["Optimization Error"].mean()
        print(summary.to_string(index=False))
        print(f"Saved outputs in {target_dir}")

    elif args.sweep == "publication":
        print("\n--- Publication Sweep ---")
        results = run_publication_sweep(dimensions=[5, 10, 20, 50], trials=args.trials, device=args.device)
        if out_dir:
            fig = plot_publication_sweep(results)
            fig.savefig(out_dir / "publication_sweep.png", dpi=150, bbox_inches="tight")
            results.to_csv(out_dir / "publication_sweep.csv", index=False)
            _save_publication_by_dimension(results, out_dir)
            print(f"Saved outputs in {out_dir}")

    elif args.sweep == "v42":
        print("\n--- Hybrid v4.2 Adaptive Benchmark ---")
        target_dir = out_dir or Path("outputs")
        results = run_v4_2_benchmark(
            dimensions=[10, 50, 100],
            max_epochs=args.steps,
            lr=args.lr,
            device=args.device,
            out_dir=target_dir,
            show=args.show,
        )
        print(results.to_string(index=False))
        print(f"Saved outputs in {target_dir}")

    else:
        print("\n--- Single Optimisation Run ---")
        histories = compare_optimizers(
            landscape=args.landscape,
            dim=args.dim,
            steps=args.steps,
            lr=args.lr,
            device=args.device,
            starts=args.starts,
        )
        for name, hist in histories.items():
            print(f"  {name:20s} final loss: {hist[-1]:.6f}")
        if out_dir:
            fig = plot_loss_curves(histories)
            fig.savefig(out_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
            _save_histories_csv(histories, out_dir / "run_histories.csv")
            print(f"Saved outputs in {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()

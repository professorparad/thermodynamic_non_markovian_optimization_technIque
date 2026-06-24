#!/usr/bin/env python3
"""
Phase 1 — Entry point for the Thermodynamic Non-Markovian Optimisation Framework.

Usage:
    python phase1/project_root/main.py --landscape rastrigin --dim 5 --steps 200
    python phase1/project_root/main.py --sweep stats --landscape ackley --trials 10
    python phase1/project_root/main.py --sweep dims --landscape sphere
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path

# Ensure project_root is on sys.path for direct execution
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.sweeps import (
    run_statistical_sweep,
    run_dimension_sweep,
    compare_optimizers,
)
from plots.visualizations import (
    plot_loss_curves,
    plot_statistical_comparison,
    plot_dimension_sweep,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thermodynamic Non-Markovian Optimisation — Phase 1")
    parser.add_argument("--landscape", type=str, default="rastrigin",
                        choices=["sphere", "rastrigin", "ackley", "rosenbrock"],
                        help="Cost landscape function")
    parser.add_argument("--dim", type=int, default=5, help="Dimensionality")
    parser.add_argument("--steps", type=int, default=200, help="Optimisation steps")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--sweep", type=str, default=None,
                        choices=["stats", "dims", "compare"],
                        help="Run a benchmark sweep instead of a single run")
    parser.add_argument("--trials", type=int, default=10, help="Trials for statistical sweep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save output plots (optional)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"=== Thermodynamic Non-Markovian Optimisation — Phase 1 ===")
    print(f"Landscape: {args.landscape} | Dim: {args.dim} | Steps: {args.steps} | LR: {args.lr}")

    if args.sweep == "stats":
        print(f"\n--- Statistical Sweep ({args.trials} trials) ---")
        results = run_statistical_sweep(
            landscape=args.landscape,
            dim=args.dim,
            steps=args.steps,
            lr=args.lr,
            n_trials=args.trials,
            seed=args.seed,
        )
        print(json.dumps(results, indent=2))

        if args.save:
            out_dir = Path(args.save)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig = plot_statistical_comparison(results)
            fig.savefig(str(out_dir / "statistical_comparison.png"), dpi=150, bbox_inches="tight")
            print(f"Plot saved to {out_dir / 'statistical_comparison.png'}")

    elif args.sweep == "dims":
        print(f"\n--- Dimension Sweep ---")
        dims = [2, 5, 10, 20, 50]
        results = run_dimension_sweep(
            landscape=args.landscape,
            dims=dims,
            steps=args.steps,
            lr=args.lr,
            n_trials=args.trials,
        )
        for d, v in sorted(results.items()):
            print(f"  dim={d:2d}: {v}")

        if args.save:
            out_dir = Path(args.save)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig = plot_dimension_sweep(results)
            fig.savefig(str(out_dir / "dimension_sweep.png"), dpi=150, bbox_inches="tight")
            print(f"Plot saved to {out_dir / 'dimension_sweep.png'}")

    elif args.sweep == "compare":
        print(f"\n--- Comparing Optimizers (single run each) ---")
        histories = compare_optimizers(
            landscape=args.landscape,
            dim=args.dim,
            steps=args.steps,
            lr=args.lr,
        )
        for name, hist in histories.items():
            print(f"  {name:20s}  final loss: {hist[-1]:.6f}")

        if args.save:
            out_dir = Path(args.save)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig = plot_loss_curves(histories)
            fig.savefig(str(out_dir / "loss_curves.png"), dpi=150, bbox_inches="tight")
            print(f"Plot saved to {out_dir / 'loss_curves.png'}")

    else:
        # Single run
        print(f"\n--- Single Optimisation Run ---")
        histories = compare_optimizers(
            landscape=args.landscape,
            dim=args.dim,
            steps=args.steps,
            lr=args.lr,
        )
        for name, hist in histories.items():
            print(f"  {name:20s}  final loss: {hist[-1]:.6f}")

        if args.save:
            out_dir = Path(args.save)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig = plot_loss_curves(histories)
            fig.savefig(str(out_dir / "loss_curves.png"), dpi=150, bbox_inches="tight")

    print("\nDone.")


if __name__ == "__main__":
    main()

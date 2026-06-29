"""Hybrid v4.2 adaptive benchmark from the pasted notebook/script.

The optimization tensors run on the selected torch device. The MPS mutual
information diagnostics are computed on CPU because quimb operates on NumPy
arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.devices import resolve_device
from src.memory import DebyeDielectric
from src.topology_mps import _build_mps, _mutual_info_matrix

Landscape = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class V42Run:
    loss_h: list[float]
    switch_epoch: int


def griewank_nd(theta: torch.Tensor) -> torch.Tensor:
    d = theta.shape[0]
    idx = torch.arange(1, d + 1, dtype=theta.dtype, device=theta.device)
    return torch.sum(theta**2) / 4000.0 - torch.prod(torch.cos(theta / torch.sqrt(idx))) + 1.0


def schwefel_nd(theta: torch.Tensor) -> torch.Tensor:
    d = theta.shape[0]
    return 418.9829 * d - torch.sum(theta * torch.sin(torch.sqrt(torch.abs(theta))))


def ackley_nd(theta: torch.Tensor) -> torch.Tensor:
    p1 = -20 * torch.exp(-0.2 * torch.sqrt(torch.mean(theta**2)))
    p2 = -torch.exp(torch.mean(torch.cos(2 * torch.pi * theta)))
    return p1 + p2 + torch.exp(theta.new_tensor(1.0)) + 20


def rastrigin_nd(theta: torch.Tensor, a: float = 10.0) -> torch.Tensor:
    d = theta.shape[0]
    return a * d + torch.sum(theta**2 - a * torch.cos(2 * torch.pi * theta))


V42_LANDSCAPES: dict[str, Landscape] = {
    "Griewank": griewank_nd,
    "Schwefel": schwefel_nd,
    "Ackley": ackley_nd,
    "Rastrigin": rastrigin_nd,
}


def run_adam_baseline(
    landscape_func: Landscape,
    dim: int,
    max_epochs: int = 800,
    lr: float = 0.01,
    device: str | torch.device = "auto",
) -> list[float]:
    torch_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    theta = torch.full((dim,), 5.0, dtype=torch.float32, device=torch_device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    loss_h: list[float] = []

    for _ in range(max_epochs):
        opt.zero_grad(set_to_none=True)
        loss = landscape_func(theta)
        loss.backward()
        opt.step()
        loss_h.append(float(loss.detach().cpu()))

    return loss_h


def run_hybrid_v4_2_adaptive(
    landscape_func: Landscape,
    dim: int,
    max_epochs: int = 800,
    lr: float = 0.01,
    switch_tol: float = 1e-5,
    device: str | torch.device = "auto",
) -> V42Run:
    torch_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    theta = torch.full((dim,), 5.0, dtype=torch.float32, device=torch_device, requires_grad=True)
    debye = DebyeDielectric(tau=10.0 * np.log1p(dim), chi=1.0 / np.sqrt(dim), dt=0.01)
    grad_h: list[float] = []
    loss_h: list[float] = []
    entropy_window: list[float] = []
    phase = "scouting"
    switch_epoch = max_epochs
    opt_adam: torch.optim.Adam | None = None

    for epoch in range(max_epochs):
        if phase == "scouting":
            loss = landscape_func(theta)
            loss.backward()
            g_tensor = theta.grad.detach()
            g_vec = g_tensor.cpu().numpy().copy()
            grad_h.append(float(np.linalg.norm(g_vec)))
            loss_h.append(float(loss.detach().cpu()))

            mean_entropy = 0.0
            mi_weight = 0.0
            if len(grad_h) >= 128:
                mps_obj = _build_mps(grad_h[-128:], bond_dim=int(12 + np.log2(dim)))
                if mps_obj is not None:
                    mean_entropy = float(np.mean([mps_obj.entropy(i) for i in range(1, mps_obj.L)]))
                    entropy_window.append(mean_entropy)

                    if len(entropy_window) > 50 and np.var(entropy_window[-30:]) < switch_tol:
                        phase = "refinement"
                        switch_epoch = epoch
                        opt_adam = torch.optim.Adam([theta], lr=lr)

                    debye.tau = 10.0 * (1.0 + 5.0 * mean_entropy)
                    mi_weight = float(np.linalg.norm(_mutual_info_matrix(mps_obj)))

            p_vec = debye.step(g_vec)
            update = g_vec + 2.0 * p_vec + (2.0 * mean_entropy + 0.1 * mi_weight) * g_vec
            update_tensor = torch.as_tensor(update, dtype=theta.dtype, device=torch_device)
            with torch.no_grad():
                theta -= lr * update_tensor
            theta.grad = None
        else:
            if opt_adam is None:
                opt_adam = torch.optim.Adam([theta], lr=lr)
            opt_adam.zero_grad(set_to_none=True)
            loss = landscape_func(theta)
            loss.backward()
            opt_adam.step()
            loss_h.append(float(loss.detach().cpu()))

    return V42Run(loss_h=loss_h, switch_epoch=switch_epoch)


def run_v4_2_benchmark(
    dimensions: list[int] | None = None,
    max_epochs: int = 800,
    lr: float = 0.01,
    device: str = "auto",
    out_dir: str | Path = "outputs",
    show: bool = False,
) -> pd.DataFrame:
    dimensions = dimensions or [10, 50, 100]
    torch_device = resolve_device(device)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_data = []
    fig, axes = plt.subplots(len(V42_LANDSCAPES), len(dimensions), figsize=(20, 18), squeeze=False)

    for i, (f_name, f_func) in enumerate(V42_LANDSCAPES.items()):
        for j, dim in enumerate(dimensions):
            print(f"Benchmarking {f_name} at {dim}D on {torch_device}...")
            hybrid = run_hybrid_v4_2_adaptive(f_func, dim, max_epochs=max_epochs, lr=lr, device=torch_device)
            adam_loss = run_adam_baseline(f_func, dim, max_epochs=max_epochs, lr=lr, device=torch_device)

            ax = axes[i, j]
            ax.plot(hybrid.loss_h, label="Hybrid v4.2 (Adaptive)", color="blue", linewidth=2)
            ax.plot(adam_loss, label="Adam Baseline", color="gray", linestyle="--", alpha=0.7)
            ax.axvline(x=hybrid.switch_epoch, color="red", linestyle=":", label="Switch Point")
            ax.set_yscale("log")
            ax.set_title(f"{f_name} {dim}D")
            ax.legend()

            report_data.append(
                {
                    "Landscape": f_name,
                    "Dim": dim,
                    "Hybrid Loss": hybrid.loss_h[-1],
                    "Adam Loss": adam_loss[-1],
                    "Switch Epoch": hybrid.switch_epoch,
                    "Device": str(torch_device),
                }
            )

    fig.tight_layout()
    fig.savefig(out_path / "hybrid_v4_2_vs_adam.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    final_report = pd.DataFrame(report_data)
    final_report.to_csv(out_path / "hybrid_v4_2_vs_adam.csv", index=False)
    return final_report

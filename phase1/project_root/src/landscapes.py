"""
Cost landscapes for benchmarking optimization algorithms.
All functions take a torch.Tensor of shape (..., D) and return a scalar loss.
"""

import torch
import numpy as np


def rastrigin(x: torch.Tensor, A: float = 10.0) -> torch.Tensor:
    """
    Rastrigin function: A * D + sum(x^2 - A * cos(2*pi*x))
    Highly multimodal, many local minima.
    """
    D = x.shape[-1]
    return A * D + torch.sum(x**2 - A * torch.cos(2 * np.pi * x), dim=-1)


def ackley(x: torch.Tensor, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> torch.Tensor:
    """
    Ackley function: wide outer basin with many sharp local minima at the centre.
    """
    D = x.shape[-1]
    sum_sq = torch.sum(x**2, dim=-1)
    sum_cos = torch.sum(torch.cos(c * x), dim=-1)
    return -a * torch.exp(-b * torch.sqrt(sum_sq / D)) - torch.exp(sum_cos / D) + a + np.e


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    """
    Rosenbrock banana function: sum_{i=1}^{D-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    Narrow valley, minimum at (1, 1, ..., 1).
    """
    return torch.sum(100 * (x[..., 1:] - x[..., :-1]**2)**2 + (1 - x[..., :-1])**2, dim=-1)


def sphere(x: torch.Tensor) -> torch.Tensor:
    """
    Simple sphere / quadratic function: sum(x^2). Convex, one global minimum at 0.
    """
    return torch.sum(x**2, dim=-1)


def schwefel(x: torch.Tensor) -> torch.Tensor:
    """
    Schwefel function: 418.9829*D - sum(x * sin(sqrt(|x|))).
    Many local minima, global minimum near (420.9687, ..., 420.9687).
    """
    D = x.shape[-1]
    return 418.9829 * D - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))), dim=-1)

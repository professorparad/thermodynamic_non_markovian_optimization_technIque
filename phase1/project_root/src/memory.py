"""
Memory / kernel functions for non-Markovian optimization.
Provides temporal kernels that bias gradient updates based on history.
"""

import torch
import numpy as np


class DebyeDielectric:
    """
    Debye dielectric relaxation kernel.
    Models a delayed response with a single relaxation time constant tau.

    K(t) = exp(-t / tau)

    Used to weight past gradients in the non-Markovian update step.
    """

    def __init__(self, tau: float = 15.0, alpha: float = 5.0):
        self.tau = tau
        self.alpha = alpha

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate kernel at time offsets t (shape: [T])."""
        return torch.exp(-t / self.tau)

    def weights(self, history_len: int, device: torch.device = None) -> torch.Tensor:
        """Return a normalized weight vector for the last `history_len` steps."""
        t = torch.arange(history_len, dtype=torch.float32, device=device).flip(0)
        w = self(t)
        return w / (w.sum() + 1e-12)


def exponential_decay_kernel(t: torch.Tensor, tau: float = 10.0) -> torch.Tensor:
    """Simple exponential decay kernel K(t) = exp(-t / tau)."""
    return torch.exp(-t / tau)


def power_law_kernel(t: torch.Tensor, exponent: float = 0.8, eps: float = 1e-6) -> torch.Tensor:
    """Power-law decay kernel K(t) = (t + eps)^{-exponent}."""
    return (t.float() + eps) ** (-exponent)

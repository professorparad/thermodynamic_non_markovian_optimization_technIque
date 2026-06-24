"""
Memory / kernel functions for non-Markovian optimization.
Provides temporal kernels that bias gradient updates based on history.
"""
# ----> old implementation
'''
import torch
import numpy as np
from typing import Optional 


class DebyeDielectric:
    """
    Debye dielectric relaxation kernel.
    Models a delayed response with a single relaxation time constant tau.

    K(t) = exp(-t / tau)

    Used to weight past gradients in the non-Markovian update step.
    """

    def __init__(self, tau: float = 15.0, chi : float = 1.0 , dt: float = 5.0):
        self.tau = tau
        self.chi = chi 
        self.dt = dt
        self._P : Optional[np.ndarray] = None 
    def step(self , E : np.ndarray):
        if self._P is None :
            self._P = np.zeros_like(E)
        self._P += self.dt * (-self._P + self.chi * E)/ self.tau
        return self._P
'''

# new implementation
import torch
import numpy as np
from typing import Optional


def exponential_decay_kernel(
    t: torch.Tensor,
    tau: float = 1.0,
):
    return torch.exp(-t / tau)


def power_law_kernel(
    t: torch.Tensor,
    exponent: float = 1.0,
    eps: float = 1.0,
):
    return 1.0 / (t + eps) ** exponent


class DebyeDielectric:
    def __init__(
        self,
        tau: float = 15.0,
        alpha: float = 1.0,
        chi: float = 1.0,
        dt: float = 5.0,
    ):
        self.tau = tau
        self.alpha = alpha
        self.chi = chi
        self.dt = dt
        self._P: Optional[np.ndarray] = None

    # Required by tests
    def __call__(self, t: torch.Tensor):
        return torch.exp(-t / self.tau)

    # Required by tests
    def weights(self, history_len: int):
        t = torch.arange(
            history_len,
            dtype=torch.float32,
        )

        w = torch.exp(-t / self.tau)
        return w / w.sum()

    # Preserve your original dynamics code
    def step(self, E: np.ndarray):
        if self._P is None:
            self._P = np.zeros_like(E)

        self._P += (
            self.dt
            * (-self._P + self.chi * E)
            / self.tau
        )

        return self._P
    
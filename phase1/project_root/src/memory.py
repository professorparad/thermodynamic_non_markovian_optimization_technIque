"""
Memory / kernel functions for non-Markovian optimization.
Provides temporal kernels that bias gradient updates based on history.
"""

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
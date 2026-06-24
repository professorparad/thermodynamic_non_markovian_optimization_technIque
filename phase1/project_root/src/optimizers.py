"""
Optimizer implementations: standard PyTorch, non-Markovian (memory-based),
and hybrid (memory + topological regularisation).
"""

import torch
import torch.optim as optim
import numpy as np

from .memory import DebyeDielectric
from .topology_mps import compute_mps_entropy, get_topo_force


class PytorchOptimizer:
    """
    Thin wrapper around a standard PyTorch optimizer (Adam by default).
    Used as the baseline in comparisons.
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-2,
        optimizer_cls: type = optim.Adam,
        **kwargs,
    ):
        self.optimizer = optimizer_cls(params, lr=lr, **kwargs)

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.optimizer.load_state_dict(state_dict)


class NonMarkovianOptimizer:
    """
    Optimizer with a non-Markovian update that convolves gradient history
    with a memory kernel (Debye dielectric relaxation).

    The effective update at step t is:
        Δθ_t = -η * Σ_{k=0}^{K} w_k * g_{t-k}
    where w_k are normalised kernel weights and g_t are past gradients.
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-2,
        memory_len: int = 20,
        tau: float = 15.0,
        alpha: float = 5.0,
    ):
        self.params = list(params)
        self.lr = lr
        self.memory_len = memory_len
        self.kernel = DebyeDielectric(tau=tau, alpha=alpha)

        # Gradient history: list of lists, one per parameter group
        self.grad_history: list[list[torch.Tensor]] = [[] for _ in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # Store current gradient
            self.grad_history[i].append(p.grad.detach().clone())
            if len(self.grad_history[i]) > self.memory_len:
                self.grad_history[i].pop(0)

            # Compute weighted sum of past gradients
            n = len(self.grad_history[i])
            if n < 2:
                effective_grad = self.grad_history[i][-1]
            else:
                w = self.kernel.weights(n, device=p.device)
                effective_grad = torch.zeros_like(p.grad)
                for k, g in enumerate(self.grad_history[i]):
                    effective_grad += w[k] * g

            p.data.add_(-self.lr * effective_grad)

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "memory_len": self.memory_len,
            "tau": self.kernel.tau,
            "alpha": self.kernel.alpha,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.lr = state_dict["lr"]
        self.memory_len = state_dict["memory_len"]
        self.kernel.tau = state_dict["tau"]
        self.kernel.alpha = state_dict["alpha"]


class HybridOptimizer:
    """
    Combines non-Markovian memory updates with topological regularisation
    from persistent homology of the parameter trajectory.

    The update is:
        Δθ = Δθ_{non-markov} + λ_topo * F_topo
    where F_topo is a random perturbation scaled by topological feature lifetimes.
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-2,
        memory_len: int = 20,
        tau: float = 15.0,
        alpha: float = 5.0,
        lambda_topo: float = 0.1,
        topo_window: int = 50,
    ):
        self.nm_optim = NonMarkovianOptimizer(
            params, lr=lr, memory_len=memory_len, tau=tau, alpha=alpha
        )
        self.lambda_topo = lambda_topo
        self.topo_window = topo_window
        self.param_trajectory: list[np.ndarray] = []
        self._step_count = 0

    def step(self) -> None:
        self.nm_optim.step()

        # Record current parameters for topology
        flat_params = torch.cat([p.data.flatten().detach().cpu() for p in self.nm_optim.params]).numpy()
        self.param_trajectory.append(flat_params)
        if len(self.param_trajectory) > self.topo_window:
            self.param_trajectory.pop(0)

        # Apply topological force every 5 steps once we have enough history
        self._step_count += 1
        if (
            self._step_count % 5 == 0
            and len(self.param_trajectory) >= 20
            and self.lambda_topo > 0
        ):
            traj_array = np.stack(self.param_trajectory)
            ndim = sum(p.numel() for p in self.nm_optim.params)
            force = get_topo_force(traj_array, lambda_topo=self.lambda_topo, ndim=ndim)

            # Distribute force across parameters
            idx = 0
            for p in self.nm_optim.params:
                numel = p.numel()
                p.data.add_(
                    torch.tensor(force[idx : idx + numel], dtype=p.dtype, device=p.device).reshape(p.shape)
                )
                idx += numel

    def zero_grad(self) -> None:
        self.nm_optim.zero_grad()

    def state_dict(self) -> dict:
        d = self.nm_optim.state_dict()
        d["lambda_topo"] = self.lambda_topo
        d["topo_window"] = self.topo_window
        d["step_count"] = self._step_count
        return d

    def load_state_dict(self, state_dict: dict) -> None:
        self.nm_optim.load_state_dict(state_dict)
        self.lambda_topo = state_dict.get("lambda_topo", self.lambda_topo)
        self.topo_window = state_dict.get("topo_window", self.topo_window)
        self._step_count = state_dict.get("step_count", 0)

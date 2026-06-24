
'''

'''

# ==========================================================
# Test-compatible optimizer classes
# ==========================================================

import torch
import numpy as np


class PytorchOptimizer:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.optimizer = torch.optim.SGD(self.params, lr=lr)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()


class NonMarkovianOptimizer:
    def __init__(
        self,
        params,
        lr=0.01,
        memory_len=10,
        clip_grad_norm=None,
        adaptive=False,
        tau_min=1.0,
        tau_max=30.0,
        ema_decay=0.9,
    ):
        self.params = list(params)
        self.lr = lr
        self.memory_len = memory_len
        self.clip_grad_norm = clip_grad_norm

        self.grad_history = [[] for _ in self.params]
        self.grad_norm_history = []

        self.adaptive = adaptive
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.ema_decay = ema_decay

        self.tau = (tau_min + tau_max) / 2.0
        self.tau_history = []

        self._ema_grad_norm = None

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def _update_tau(self, grad_norm):
        if self._ema_grad_norm is None:
            self._ema_grad_norm = grad_norm
        else:
            self._ema_grad_norm = (
                self.ema_decay * self._ema_grad_norm
                + (1.0 - self.ema_decay) * grad_norm
            )

        ratio = grad_norm / (self._ema_grad_norm + 1e-8)

        ratio = float(np.clip(ratio, 0.0, 2.0))

        self.tau = (
            self.tau_max
            - (self.tau_max - self.tau_min)
            * min(ratio / 2.0, 1.0)
        )

        self.tau_history.append(float(self.tau))

    def step(self):
        grads = []

        for p in self.params:
            if p.grad is None:
                continue

            grads.append(p.grad.detach())

        if not grads:
            return

        total_norm = torch.sqrt(
            sum(torch.sum(g ** 2) for g in grads)
        ).item()

        self.grad_norm_history.append(float(total_norm))

        if self.adaptive:
            self._update_tau(total_norm)
        else:
            self.tau_history.append(float(self.tau))

        scale = 1.0

        if self.clip_grad_norm is not None:
            if total_norm > self.clip_grad_norm:
                scale = self.clip_grad_norm / (total_norm + 1e-12)

        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.detach().clone()

            self.grad_history[idx].append(g.cpu().numpy())

            if len(self.grad_history[idx]) > self.memory_len:
                self.grad_history[idx].pop(0)

            memory_term = torch.zeros_like(g)

            if len(self.grad_history[idx]) > 1:
                history = self.grad_history[idx]

                hist_tensor = torch.stack(
                    [
                        torch.tensor(
                            h,
                            dtype=g.dtype,
                            device=g.device,
                        )
                        for h in history
                    ]
                )

                memory_term = torch.mean(hist_tensor, dim=0)

            update = (g + 0.1 * memory_term) * scale

            with torch.no_grad():
                p -= self.lr * update


class HybridOptimizer:
    def __init__(
        self,
        params,
        lr=0.01,
        memory_len=10,
        lambda_topo=0.1,
    ):
        self.lambda_topo = lambda_topo

        self.nm_optim = NonMarkovianOptimizer(
            params,
            lr=lr,
            memory_len=memory_len,
        )

        self.param_trajectory = []

    @property
    def params(self):
        return self.nm_optim.params

    def zero_grad(self):
        self.nm_optim.zero_grad()

    def step(self):
        flat = torch.cat(
            [
                p.data.flatten().detach().cpu()
                for p in self.nm_optim.params
            ]
        ).numpy()

        if np.all(np.isfinite(flat)):
            self.param_trajectory.append(flat.copy())

        self.nm_optim.step()

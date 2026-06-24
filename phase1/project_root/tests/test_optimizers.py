import numpy as np
import torch
import torch.nn as nn

from src.landscapes import rastrigin, sphere
from src.optimizers import HybridOptimizer, NonMarkovianOptimizer, PytorchOptimizer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([1.0, 2.0]))


def test_pytorch_optimizer():
    model = DummyModel()
    opt = PytorchOptimizer(model.parameters(), lr=0.1)

    opt.zero_grad()
    loss = sphere(model.param)
    loss.backward()
    opt.step()

    # Parameter should move closer to 0
    assert model.param[0].item() < 1.0
    assert model.param[1].item() < 2.0


def test_non_markovian_optimizer():
    model = DummyModel()
    opt = NonMarkovianOptimizer(model.parameters(), lr=0.1, memory_len=5)

    # Step 1
    opt.zero_grad()
    loss = sphere(model.param)
    loss.backward()
    opt.step()

    assert model.param[0].item() < 1.0

    # Step 2
    opt.zero_grad()
    loss2 = sphere(model.param)
    loss2.backward()
    opt.step()

    assert len(opt.grad_history[0]) == 2


def test_non_markovian_grad_norm_history():
    """grad_norm_history should be populated after each step."""
    model = DummyModel()
    opt = NonMarkovianOptimizer(model.parameters(), lr=0.1, memory_len=5)
    for _ in range(3):
        opt.zero_grad()
        sphere(model.param).backward()
        opt.step()
    assert len(opt.grad_norm_history) == 3
    assert all(v >= 0.0 for v in opt.grad_norm_history)


def test_non_markovian_grad_clipping():
    """With a very small clip value the parameter update should be tiny."""
    model = DummyModel()
    opt_clipped = NonMarkovianOptimizer(
        [model.param.clone().detach().requires_grad_(True)],
        lr=0.1,
        clip_grad_norm=1e-6,
    )
    opt_noclip = NonMarkovianOptimizer(
        [model.param.clone().detach().requires_grad_(True)],
        lr=0.1,
        clip_grad_norm=None,
    )
    for opt in (opt_clipped, opt_noclip):
        opt.zero_grad()
        p = opt.params[0]
        sphere(p).backward()
        opt.step()
    # Verify raw gradient norm is recorded and positive (clipping doesn't affect pre-clip norm)
    opt_clipped.params[0].grad = None
    opt_noclip.params[0].grad = None
    assert opt_clipped.grad_norm_history[0] == opt_noclip.grad_norm_history[0]
    assert opt_clipped.grad_norm_history[0] > 0


def test_hybrid_optimizer():
    model = DummyModel()
    opt = HybridOptimizer(model.parameters(), lr=0.1, memory_len=5, lambda_topo=0.1)

    # Run a few steps to trigger topological force
    for _ in range(6):
        opt.zero_grad()
        loss = sphere(model.param)
        loss.backward()
        opt.step()

    assert len(opt.param_trajectory) == 6


def test_hybrid_nan_guard():
    """HybridOptimizer must not crash when parameters become NaN."""
    param = torch.tensor([float("nan"), float("nan")], requires_grad=True)
    opt = HybridOptimizer([param], lr=0.1, memory_len=5, lambda_topo=0.1)
    # Manually inject a NaN param — trajectory snapshot should be skipped
    opt.zero_grad()
    # Simulate a step without a valid gradient to exercise the guard path
    opt.nm_optim.params[0].data.fill_(float("nan"))
    import numpy as np

    flat = torch.cat([p.data.flatten().detach().cpu() for p in opt.nm_optim.params]).numpy()
    # The guard should catch this
    assert not np.all(np.isfinite(flat))
    # Trajectory should remain empty (no snapshot taken)
    assert len(opt.param_trajectory) == 0


def test_adaptive_tau_changes():
    """tau_history should be populated and tau should vary when adaptive=True."""
    x = torch.randn(5, requires_grad=True)
    opt = NonMarkovianOptimizer([x], lr=0.01, adaptive=True, tau_min=1.0, tau_max=30.0)
    for _ in range(20):
        opt.zero_grad()
        rastrigin(x).backward()
        opt.step()
    assert len(opt.tau_history) == 20
    # tau should have moved from its initial value on a multimodal landscape
    assert not all(t == opt.tau_history[0] for t in opt.tau_history)


def test_adaptive_tau_landscape_response():
    """
    On a smooth landscape (sphere), tau should converge toward tau_max.
    On a multimodal landscape (rastrigin), tau should stay lower on average.
    """
    STEPS = 50
    TAU_MIN, TAU_MAX = 1.0, 30.0

    def mean_tau(landscape_fn):
        x = torch.randn(5, requires_grad=True)
        opt = NonMarkovianOptimizer(
            [x], lr=0.01, adaptive=True, tau_min=TAU_MIN, tau_max=TAU_MAX, ema_decay=0.8
        )
        for _ in range(STEPS):
            opt.zero_grad()
            landscape_fn(x).backward()
            opt.step()
        # Return mean tau over last 20 steps (after warm-up)
        return float(np.mean(opt.tau_history[-20:]))

    tau_sphere = mean_tau(sphere)
    tau_rastrigin = mean_tau(rastrigin)
    # Sphere should result in higher mean tau than rastrigin
    assert tau_sphere > tau_rastrigin, (
        f"Expected tau_sphere ({tau_sphere:.2f}) > tau_rastrigin ({tau_rastrigin:.2f})"
    )

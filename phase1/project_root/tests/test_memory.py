import torch

from src.memory import DebyeDielectric, exponential_decay_kernel, power_law_kernel


def test_debye_dielectric():
    kernel = DebyeDielectric(tau=15.0, alpha=5.0)
    t = torch.tensor([0.0, 15.0, 30.0])
    res = kernel(t)

    # K(0) = 1, K(15) = e^-1, K(30) = e^-2
    assert torch.isclose(res[0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(res[1], torch.tensor(0.367879), atol=1e-5)

    # weights check
    w = kernel.weights(history_len=5)
    assert w.shape == (5,)
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5)


def test_exponential_decay_kernel():
    t = torch.tensor([0.0, 10.0])
    res = exponential_decay_kernel(t, tau=10.0)
    assert torch.isclose(res[0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(res[1], torch.tensor(0.367879), atol=1e-5)


def test_power_law_kernel():
    t = torch.tensor([0.0, 1.0])
    res = power_law_kernel(t, exponent=1.0, eps=1.0)
    assert torch.isclose(res[0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(res[1], torch.tensor(0.5), atol=1e-5)

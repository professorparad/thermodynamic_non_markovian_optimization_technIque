import torch

from src.landscapes import ackley, griewank, levy, rastrigin, rosenbrock, schwefel, sphere


def test_rastrigin():
    # Global minimum at origin is 0
    x = torch.zeros(3)
    loss = rastrigin(x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    # Dimensionality check
    x_batch = torch.zeros(5, 3)
    loss_batch = rastrigin(x_batch)
    assert loss_batch.shape == (5,)


def test_ackley():
    # Global minimum at origin is 0
    x = torch.zeros(3)
    loss = ackley(x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_rosenbrock():
    # Minimum is at (1, 1, ...)
    x = torch.ones(4)
    loss = rosenbrock(x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_sphere():
    # Minimum is at origin
    x = torch.zeros(2)
    loss = sphere(x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_schwefel():
    # Global minimum is near 420.9687
    x = torch.full((3,), 420.968746)
    loss = schwefel(x)
    assert loss < 1e-3  # Near zero


def test_levy():
    # Global minimum is at x = (1, 1, ...)
    x = torch.ones(4)
    loss = levy(x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    # Batch shape check
    x_batch = torch.ones(5, 4)
    loss_batch = levy(x_batch)
    assert loss_batch.shape == (5,)


def test_griewank():
    # Global minimum is at origin
    x = torch.zeros(4)
    loss = griewank(x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    # Batch shape check
    x_batch = torch.zeros(5, 4)
    loss_batch = griewank(x_batch)
    assert loss_batch.shape == (5,)

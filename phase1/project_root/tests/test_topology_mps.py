import numpy as np

from src.topology_mps import compute_mps_entropy, compute_persistent_homology, get_topo_force


def test_compute_mps_entropy():
    # Simple trajectory
    trajectory = np.random.rand(16)
    entropy = compute_mps_entropy(trajectory, bond_dim=2)
    assert isinstance(entropy, float)
    assert entropy >= 0.0


def test_compute_persistent_homology():
    trajectory = np.random.rand(10, 2)
    res = compute_persistent_homology(trajectory, maxdim=1)
    assert "dgms" in res
    assert len(res["dgms"]) >= 2


def test_get_topo_force():
    trajectory = np.random.rand(10, 2)
    force = get_topo_force(trajectory, lambda_topo=0.1, window=5, ndim=2)
    assert force.shape == (2,)

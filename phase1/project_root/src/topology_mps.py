"""
Topological and MPS (Matrix Product State) utilities.
Compute persistent homology and MPS entanglement entropy from optimization trajectories.
"""

import torch
import numpy as np
from ripser import ripser
import quimb.tensor as qtn


def compute_mps_entropy(
    history: list | np.ndarray,
    bond_dim: int = 32,
) -> float:
    """
    Convert a 1D trajectory history into a Matrix Product State and compute
    the average entanglement entropy across bipartitions.

    Parameters
    ----------
    history : list or np.ndarray
        Flattenable 1D trajectory of loss / parameter values.
    bond_dim : int
        Maximum bond dimension for MPS compression.

    Returns
    -------
    float
        Mean entanglement entropy.
    """
    x = np.array(history).flatten()
    n = int(np.floor(np.log2(len(x))))
    if n < 1:
        return 0.0
    vec = x[: 2**n]
    mps = qtn.MatrixProductState.from_dense(
        vec / (np.linalg.norm(vec) + 1e-12), [2] * n
    )
    mps.compress(max_bond=bond_dim)
    return float(np.mean([mps.entropy(i) for i in range(1, mps.L)]))


def compute_persistent_homology(
    trajectory: np.ndarray,
    maxdim: int = 1,
) -> dict:
    """
    Compute persistent homology of a trajectory point cloud.

    Parameters
    ----------
    trajectory : np.ndarray of shape (T, D)
        Sequence of D-dimensional points visited during optimisation.
    maxdim : int
        Maximum homology dimension to compute.

    Returns
    -------
    dict
        ripser output dictionary with keys 'dgms' (list of diagrams).
    """
    return ripser(trajectory, maxdim=maxdim)


def get_topo_force(
    trajectory: np.ndarray,
    lambda_topo: float = 0.1,
    window: int = 50,
    ndim: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Compute a topological force that pushes the optimiser away from
    topologically simple regions (low persistent homology lifetimes).

    Parameters
    ----------
    trajectory : np.ndarray of shape (T, D)
    lambda_topo : float
        Strength of the topological regularisation.
    window : int
        Number of most recent points to use.
    ndim : int
        Dimensionality of the force vector (matches parameter space).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (ndim,)
        Random perturbation weighted by topological lifetimes.
    """
    pts = np.array(trajectory[-window:])
    dgms = ripser(pts, maxdim=1)["dgms"]
    if len(dgms) > 1 and len(dgms[1]) > 0:
        lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
        rng = np.random.default_rng(seed)
        return lambda_topo * float(np.sum(lifetimes)) * rng.standard_normal(ndim)
    return np.zeros(ndim)

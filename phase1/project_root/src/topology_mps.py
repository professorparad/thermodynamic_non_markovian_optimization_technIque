"""
Topological and MPS (Matrix Product State) utilities.
Compute persistent homology and MPS entanglement entropy from optimization trajectories.
"""
import torch
import warnings 
from typing import List  , Optional 
import numpy as np
try : 
    import quimb.tensor as qtn 
    HAS_QUIMB = True 
except ImportError : 
    HAS_QUIMB = False 
    warnings.warn("quimb not found " , stacklevel = 2)
try : 
    from ripser import ripser
    has_risper = True
    
except ImportError :
    has_risper = False 
    warnings.warn("ripser not found; topology-escape kicks disabled.", stacklevel=2)

def _build_mps (history : List[float] , bond_dim : int = 12):
    if not HAS_QUIMB : return None 
    x = np.asarray(history , dtype = float)
    n = int(np.floor(np.log2(len(x))))
    if n < 1 : return None 
    x = x[: 2 ** n ]
    norm = np.linalg.norm(x)
    if norm < 1e-12 : return None 
    x = x[: 2 ** n ]
    norm = np.linalg.norm(x)
    if norm < 1e-12 : return None
    mps = qtn.MatrixProductState.from_dense(x / norm , [2] * n )
    return mps 

def _mps_bond_entropy(mps):
    if mps is None : return 0.0 
    ents = []
    for i in range(1 , mps.L):
        sv = mps.schmidt_values(i)
        p = sv[sv > 1e-12 ]**2
        p /= p.sum()
        ents.append(float(-np.dot(p , np.log2(p))))
    return float(np.mean(ents)) if ents else 0.0 

def _mean_bond_entropy(mps):
    return _mps_bond_entropy(mps)
    
def _mutual_info_matrix(mps):
        if mps is None : return np.zeros((1 , 1 ))
        L = mps.L 
        mim = np.zeros((L , L ))
        return mim 

def _topo_kick(current , pts , current_loss , lambda_topo ):
        if not has_risper or len(pts) <15 : return np.zeros_like(current) , 0.0 

        dgms = ripser(pts , maxdim = 1 , distance_matrix = False)["dgms"]

        if len(dgms)< 2 or len(dgms[1]) == 0 : return np.zeros_like(current) , 0.0

        lifetimes = dgms[1][: , 1] -dgms[1][: , 0]
        total_p = float(np.sum(lifetimes))
        return np.zeros_like(current) , total_p



# new additions

def compute_mps_entropy(
    trajectory: np.ndarray,
    bond_dim: int = 12,
) -> float:
    """
    Compute mean MPS bond entropy from a trajectory.
    """

    mps = _build_mps(list(np.asarray(trajectory).ravel()), bond_dim=bond_dim)

    if mps is None:
        return 0.0

    return float(_mean_bond_entropy(mps))


def compute_persistent_homology(
    trajectory: np.ndarray,
    maxdim: int = 1,
):
    """
    Compute persistent homology using ripser.
    """

    trajectory = np.asarray(trajectory)

    if has_risper:
        return ripser(
            trajectory,
            maxdim=maxdim,
            distance_matrix=False,
        )

    # fallback when ripser is unavailable
    return {
        "dgms": [
            np.empty((0, 2)),
            np.empty((0, 2)),
        ]
    }


def get_topo_force(
    trajectory: np.ndarray,
    lambda_topo: float = 0.1,
    window: int = 5,
    ndim: int = 2,
):
    """
    Compute topology-based force term.
    """

    trajectory = np.asarray(trajectory)

    if len(trajectory) == 0:
        return np.zeros(ndim)

    recent = trajectory[-window:]

    current = recent[-1]

    force, _ = _topo_kick(
        current=current,
        pts=recent,
        current_loss=0.0,
        lambda_topo=lambda_topo,
    )

    force = np.asarray(force)

    if force.shape != (ndim,):
        force = np.zeros(ndim)

    return force

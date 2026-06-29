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
    HAS_RIPSER = True
except ImportError :
    ripser = None
    HAS_RIPSER = False 

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


def _reduced_rho(mps, sites):
        return mps.partial_trace_to_mpo(sites).to_dense()


def _von_neumann_entropy(rho, tol=1e-12):
        evals = np.linalg.eigvalsh(rho)
        evals = evals[evals > tol]
        return float(-np.sum(evals * np.log2(evals)))
    
def _mutual_info_matrix(mps):
        if mps is None : return np.zeros((1 , 1 ))
        L = mps.L 
        mim = np.zeros((L , L ))
        single_entropies = np.array([
            _von_neumann_entropy(_reduced_rho(mps, [i]))
            for i in range(L)
        ])
        for i in range(L):
            for j in range(i + 1, L):
                rho_ij = _reduced_rho(mps, [i, j])
                mi = single_entropies[i] + single_entropies[j] - _von_neumann_entropy(rho_ij)
                mim[i, j] = mim[j, i] = max(0.0, mi)
        return mim 

def _topo_kick(current , pts , current_loss , lambda_topo ):
        if not HAS_RIPSER:
            warnings.warn("ripser not found; topology-escape kicks disabled.", stacklevel=2)
            return np.zeros_like(current) , 0.0
        if len(pts) <15 : return np.zeros_like(current) , 0.0 

        dgms = ripser(pts , maxdim = 1 , distance_matrix = False)["dgms"]

        if len(dgms)< 2 or len(dgms[1]) == 0 : return np.zeros_like(current) , 0.0

        lifetimes = dgms[1][: , 1] -dgms[1][: , 0]
        total_p = float(np.sum(lifetimes))
        return np.zeros_like(current) , total_p

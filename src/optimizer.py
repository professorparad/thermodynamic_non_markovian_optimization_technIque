
import torch
import numpy as np
import quimb.tensor as qtn
from ripser import ripser

class HybridOptimizerV39:
    def __init__(self, tau0=15.0, alpha=5.0, lambda_topo=0.1):
        self.tau0 = tau0
        self.alpha = alpha
        self.lambda_topo = lambda_topo

    def get_mps_entropy(self, history, bond_dim=32):
        x = np.array(history).flatten()
        n = int(np.floor(np.log2(len(x))))
        if n < 1: return 0.0
        vec = x[:2**n]
        mps = qtn.MatrixProductState.from_dense(vec / (np.linalg.norm(vec) + 1e-12), [2]*n)
        mps.compress(max_bond=bond_dim)
        return np.mean([mps.entropy(i) for i in range(1, mps.L)])

    def get_topo_force(self, traj, ndim):
        pts = np.array(traj[-50:])
        dgms = ripser(pts, maxdim=1)['dgms']
        if len(dgms) > 1 and len(dgms[1]) > 0:
            lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
            return self.lambda_topo * np.sum(lifetimes) * np.random.randn(ndim)
        return np.zeros(ndim)

import torch
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from typing import List

from .memory import DebyeDielectric
from .topology_mps import _build_mps, _mean_bond_entropy, _topo_kick
from .landscapes import cost

@dataclass
class RunResult:
    method: str
    traj: np.ndarray
    loss_h: List[float]
    kicks: List[int] = field(default_factory=list)
    mim_evo: List[np.ndarray] = field(default_factory=list)

def run_pytorch_optimizer(name, theta0, target_func=cost, epochs=600, lr=0.015):
    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    if name.lower() == 'adam':
        optimizer = optim.Adam([theta], lr=lr)
    elif name.lower() == 'lbfgs':
        optimizer = optim.LBFGS([theta], lr=lr)
    
    loss_h, traj = [], []
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            loss = target_func(theta)
            loss.backward()
            return loss
        
        if name.lower() == 'lbfgs':
            loss = optimizer.step(closure)
        else:
            loss = target_func(theta)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        loss_h.append(loss.item())
        traj.append(theta.detach().numpy().copy())
        
    return RunResult(name, np.array(traj), loss_h)

def run_comprehensive_optimizer(mode='mim_topo', target_func=cost, epochs=600, lr=0.015, alpha=5.0, theta_init=None):
    if theta_init is None:
        theta_init = [2.5, 1.5]
    theta = torch.tensor(theta_init, dtype=torch.float64, requires_grad=True)
    ndim = len(theta_init)
    debye = DebyeDielectric(tau=15.0)

    grad_h, loss_h, traj, kicks = [], [], [], []

    for epoch in range(epochs):
        loss = target_func(theta)
        loss.backward()
        g_vec = theta.grad.detach().numpy().copy()

        grad_h.append(g_vec)
        loss_h.append(loss.item())
        traj.append(theta.detach().numpy().copy())

        mi_kernel_val = 0.0
        if mode == 'mim_topo' and len(grad_h) >= 100:
            window = np.array(grad_h[-100:]).flatten()
            mps_obj = _build_mps(window, bond_dim=32)
            if mps_obj:
                ent = _mean_bond_entropy(mps_obj)
                debye.tau = 25.0 * (1.0 + ent)
                mi_kernel_val = 0.0 # Placeholder for actual get_mim_weight

        topo_force = np.zeros(ndim)
        if mode == 'mim_topo' and epoch > 50 and epoch < (epochs - 100) and epoch % 25 == 0:
            pts = np.array(traj[-50:])
            # Topo kick simulation inline based on your code
            try:
                from ripser import ripser
                dgms = ripser(pts, maxdim=1, distance_matrix=False)["dgms"]
                if len(dgms) > 1 and len(dgms[1]) > 0:
                    lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
                    if np.max(lifetimes) > 0.015:
                        kicks.append(epoch)
                        decay = max(0.1, 1.0 - (epoch / epochs))
                        topo_force = 3.0 * decay * np.sum(lifetimes) * np.random.randn(ndim)
            except ImportError:
                pass

        P = debye.step(g_vec)
        update = g_vec + alpha * P + (0.4 * mi_kernel_val) * g_vec + topo_force

        with torch.no_grad():
            theta -= lr * torch.tensor(update)
        theta.grad.zero_()

    return RunResult(mode, np.array(traj), loss_h, kicks)

def run_hybrid_v3_9(target_func, ndim=5, scout_epochs=400, refine_epochs=800, lr=0.02):
    # SCOUTING PHASE
    theta_init = np.random.uniform(-5, 5, size=(ndim,)).tolist()
    scout_res = run_comprehensive_optimizer(
        mode='mim_topo',
        target_func=target_func,
        epochs=scout_epochs,
        lr=lr,
        theta_init=theta_init,
    )
    
    # ELITE SELECTION
    best_idx = np.argmin(scout_res.loss_h)
    theta_elite = scout_res.traj[best_idx]
    
    # REFINEMENT PHASE
    theta = torch.tensor(theta_elite, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    refine_loss, refine_traj = [], []
    for _ in range(refine_epochs):
        optimizer.zero_grad()
        l = target_func(theta)
        l.backward()
        optimizer.step()
        scheduler.step()
        
        refine_loss.append(l.item())
        refine_traj.append(theta.detach().numpy().copy())
        
    return RunResult("Hybrid v3.9", 
                     np.concatenate([scout_res.traj, np.array(refine_traj)]),
                     scout_res.loss_h + refine_loss,
                     kicks=scout_res.kicks)

import torch
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from typing import List

from .memory import DebyeDielectric
from .topology_mps import _build_mps, _mean_bond_entropy, _topo_kick
from .landscapes import cost
from .devices import resolve_device

@dataclass
class RunResult:
    method: str
    traj: np.ndarray
    loss_h: List[float]
    kicks: List[int] = field(default_factory=list)
    mim_evo: List[np.ndarray] = field(default_factory=list)

def run_pytorch_optimizer(name, theta0, target_func=cost, epochs=600, lr=0.015, device="auto"):
    torch_device = resolve_device(device)
    theta = torch.tensor(theta0, dtype=torch.float64, device=torch_device, requires_grad=True)
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
        traj.append(theta.detach().cpu().numpy().copy())
        
    return RunResult(name, np.array(traj), loss_h)

def run_comprehensive_optimizer(mode='mim_topo', target_func=cost, epochs=600, lr=0.015, alpha=5.0, theta_init=None, device="auto"):
    if theta_init is None:
        theta_init = [2.5, 1.5]
    torch_device = resolve_device(device)
    theta = torch.tensor(theta_init, dtype=torch.float64, device=torch_device, requires_grad=True)
    ndim = len(theta_init)
    debye = DebyeDielectric(tau=15.0)

    grad_h, loss_h, traj, kicks = [], [], [], []

    for epoch in range(epochs):
        loss = target_func(theta)
        loss.backward()
        g_vec = theta.grad.detach().cpu().numpy().copy()

        grad_h.append(g_vec)
        loss_h.append(loss.item())
        traj.append(theta.detach().cpu().numpy().copy())

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
            theta -= lr * torch.as_tensor(update, dtype=theta.dtype, device=torch_device)
        theta.grad.zero_()

    return RunResult(mode, np.array(traj), loss_h, kicks)

def run_hybrid_v3_9(target_func, ndim=5, scout_epochs=400, refine_epochs=800, lr=0.02, device="auto"):
    # SCOUTING PHASE
    theta_init = np.random.uniform(-5, 5, size=(ndim,)).tolist()
    scout_res = run_comprehensive_optimizer(
        mode='mim_topo',
        target_func=target_func,
        epochs=scout_epochs,
        lr=lr,
        theta_init=theta_init,
        device=device,
    )
    
    # ELITE SELECTION
    best_idx = np.argmin(scout_res.loss_h)
    theta_elite = scout_res.traj[best_idx]
    
    # REFINEMENT PHASE
    torch_device = resolve_device(device)
    theta = torch.tensor(theta_elite, dtype=torch.float64, device=torch_device, requires_grad=True)
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
        refine_traj.append(theta.detach().cpu().numpy().copy())
        
    return RunResult("Hybrid v3.9", 
                     np.concatenate([scout_res.traj, np.array(refine_traj)]),
                     scout_res.loss_h + refine_loss,
                     kicks=scout_res.kicks)


def run_global_hybrid_optimizer(
    target_func=cost,
    ndim=5,
    scout_epochs=300,
    refine_epochs=300,
    lr=0.02,
    starts=8,
    init_range=5.0,
    noise_scale=0.08,
    memory_alpha=1.25,
    device="auto",
):
    """Multi-start non-Markovian scout followed by Adam refinement.

    This is designed for multimodal global-minimum searches. It explores many
    basins in parallel, uses a decaying Langevin-style noise term to escape
    shallow traps, keeps the best point seen by any scout, then lets Adam refine
    that elite candidate.
    """
    torch_device = resolve_device(device)
    dtype = torch.float64
    starts = max(1, int(starts))
    scout_epochs = max(1, int(scout_epochs))
    refine_epochs = max(1, int(refine_epochs))

    theta = (
        torch.rand((starts, ndim), dtype=dtype, device=torch_device) * 2.0 - 1.0
    ) * float(init_range)
    theta.requires_grad_(True)
    memory = torch.zeros_like(theta)
    best_theta = theta[0].detach().clone()
    best_loss = float("inf")
    loss_h, traj, kicks = [], [], []
    no_improve = 0

    for epoch in range(scout_epochs):
        if theta.grad is not None:
            theta.grad = None

        losses = torch.stack([target_func(theta[i]) for i in range(starts)])
        total_loss = torch.sum(losses)
        total_loss.backward()

        with torch.no_grad():
            min_idx = int(torch.argmin(losses).detach().cpu())
            current_best = float(losses[min_idx].detach().cpu())
            if current_best + 1e-12 < best_loss:
                best_loss = current_best
                best_theta = theta[min_idx].detach().clone()
                no_improve = 0
            else:
                no_improve += 1

            grad = theta.grad.detach()
            grad_norm = torch.linalg.vector_norm(grad, dim=1, keepdim=True).clamp_min(1e-12)
            grad_unit = grad / grad_norm
            temperature = noise_scale * (1.0 - epoch / scout_epochs) ** 1.5
            memory.mul_(0.88).add_(grad_unit, alpha=0.12)
            noise = torch.randn_like(theta) * temperature
            theta.sub_(lr * (grad + memory_alpha * memory)).add_(noise)
            theta.clamp_(-float(init_range), float(init_range))

            if no_improve >= 40 and starts > 1:
                worst_idx = int(torch.argmax(losses).detach().cpu())
                theta[worst_idx].copy_(best_theta + torch.randn(ndim, dtype=dtype, device=torch_device) * temperature * 2.0)
                theta[worst_idx].clamp_(-float(init_range), float(init_range))
                memory[worst_idx].zero_()
                kicks.append(epoch)
                no_improve = 0

            loss_h.append(best_loss)
            traj.append(best_theta.detach().cpu().numpy().copy())

    theta_refine = best_theta.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta_refine], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=refine_epochs, eta_min=lr * 0.05)

    for _ in range(refine_epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = target_func(theta_refine)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = float(loss.detach().cpu())
        if current_loss < best_loss:
            best_loss = current_loss
            best_theta = theta_refine.detach().clone()

        loss_h.append(best_loss)
        traj.append(best_theta.detach().cpu().numpy().copy())

    return RunResult("Global Hybrid", np.array(traj), loss_h, kicks=kicks)

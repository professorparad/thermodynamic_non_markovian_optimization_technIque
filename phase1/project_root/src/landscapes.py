"""
Cost landscapes for benchmarking optimization algorithms.
All functions take a torch.Tensor of shape (..., D) and return a scalar loss.
"""
import torch
import numpy as np

def cost(theta : torch.tensor):
    bowl = torch.sum(theta**2 + 2.0 *torch.sin(5.0 * theta ))
    outer = torch.outer(theta , theta )
    mask = torch.triu(torch.ones(len(theta) , len(theta ) , dtype = torch.bool ,  device = theta.device ) , diagonal = 1 )
    coupling = torch.sum(torch.cos(10.0 * outer[mask])) / len(theta)
    return bowl+ coupling
def cost_np(x: np.ndarray):
    return cost(torch.tensor(x , dtype = torch.float)).item()
def rastrigin(theta : torch.Tensor):
    A = 10 
    return A * len(theta) + torch.sum(theta**2 - A* torch.cos(2*np.pi*theta))
def ackley(theta : torch.Tensor):
    a , b , c = 20 , 0.2  , 2* np.pi 
    d = len(theta )
    sum1 = torch.sum(theta**2)
    sum2 = torch.sum(torch.cos(c*theta))
    term1 = -a * torch.exp(-b* torch.sqrt(sum1 / d ))
    term2 = -torch.exp(sum2/ d)
    return term1 + term2 + a + np.e
def schwefel(theta : torch.Tensor): 
    a , b , c = 20 , 0.2 , 2*np.pi 
    d = len(theta)
    return 418.9829 * d - torch.sum(theta * torch.sin(torch.sqrt(torch.abs(theta))))
def griewank(theta : torch.Tensor):
    sum_part = torch.sum(theta**2)/4000
    idx = torch.arange(1, len(theta) + 1, dtype=theta.dtype, device=theta.device)
    prod_part = torch.prod(torch.cos(theta / torch.sqrt(idx)))
    return sum_part - prod_part + 1 
def levy(theta : torch.Tensor) : 
    w = 1 + (theta - 1 ) / 4 
    term1 = torch.sin(np.pi * w[0]) ** 2
    term3 = ((w[-1] - 1 ) ** 2 * (1+ torch.sin(2 * np.pi * w[: -1]+ 1 )))
    middle_sum = torch.sum((w[:1] - 1) **2 * (1+ 10 * torch.sin(np.pi * w[:1] +1) **2 ))
    return term1 + middle_sum+term3 

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class KineticNet(nn.Module):
    """
    Learns kinetic energy T(θ̇).
    Input: θ̇ only
    Output: scalar T
    
    Enforces T > 0 by outputting (1/2) * MLP(θ̇)² 
    which mimics (1/2) * M(θ) * θ̇²
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Args:
            input_dim: number of velocity states (1 for single, 2 for double)
            hidden_dim: hidden layer size
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),  # output same dim as input
        )
    
    def forward(self, q_dot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_dot: velocity tensor, shape (batch, input_dim)
            
        Returns:
            T: kinetic energy, shape (batch, 1)
        """
        # (1/2) * ||MLP(θ̇)||² ensures T > 0
        h = self.net(q_dot)
        T = 0.5 * (h * h).sum(dim=-1, keepdim=True)
        return T


class PotentialNet(nn.Module):
    """
    Learns potential energy V(θ).
    Input: θ only
    Output: scalar V
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Args:
            input_dim: number of position states (1 for single, 2 for double)
            hidden_dim: hidden layer size
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: position tensor, shape (batch, input_dim)
            
        Returns:
            V: potential energy, shape (batch, 1)
        """
        return self.net(q)


class LagrangianNN(nn.Module):
    """
    Full Lagrangian Neural Network.
    
    L(θ, θ̇) = KineticNet(θ̇) - PotentialNet(θ)
    
    Equations of motion derived via Euler-Lagrange:
    d/dt(∂L/∂θ̇) - ∂L/∂θ = 0
    """
    
    def __init__(
        self,
        n_dof: int,
        hidden_dim: int = 64,
    ):
        """
        Args:
            n_dof: degrees of freedom (1 for single, 2 for double pendulum)
            hidden_dim: hidden layer size
        """
        super().__init__()
        
        self.n_dof = n_dof
        self.T_net = KineticNet(n_dof, hidden_dim)
        self.V_net = PotentialNet(n_dof, hidden_dim)
    
    def lagrangian(
        self,
        q: torch.Tensor,
        q_dot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Lagrangian L = T - V.
        
        Args:
            q: positions, shape (batch, n_dof)
            q_dot: velocities, shape (batch, n_dof)
            
        Returns:
            L: Lagrangian, shape (batch, 1)
        """
        T = self.T_net(q_dot)
        V = self.V_net(q)
        return T - V
    
    def forward(
        self,
        q: torch.Tensor,
        q_dot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute accelerations q̈ from Euler-Lagrange equations.
        
        d/dt(∂L/∂q̇) - ∂L/∂q = 0
        → M(q)*q̈ = ∂L/∂q - (∂²L/∂q̇∂q)*q̇
        
        Args:
            q: positions, shape (batch, n_dof)
            q_dot: velocities, shape (batch, n_dof)
            
        Returns:
            q_ddot: accelerations, shape (batch, n_dof)
        """
        # Enable gradients
        q = q.requires_grad_(True)
        q_dot = q_dot.requires_grad_(True)
        
        # Compute Lagrangian
        L = self.lagrangian(q, q_dot)
        L_sum = L.sum()
        
        # ∂L/∂q̇ — shape (batch, n_dof)
        dL_dqdot = torch.autograd.grad(
            L_sum, q_dot,
            create_graph=True,
        )[0]
        
        # ∂L/∂q — shape (batch, n_dof)
        dL_dq = torch.autograd.grad(
            L_sum, q,
            create_graph=True,
        )[0]
        
        # d/dt(∂L/∂q̇) = (∂²L/∂q̇²)*q̈ + (∂²L/∂q̇∂q)*q̇
        # Solve for q̈: M*q̈ = ∂L/∂q - (∂²L/∂q̇∂q)*q̇
        # where M = ∂²L/∂q̇²  (mass matrix)
        
        # Build mass matrix M = ∂²L/∂q̇²
        M = torch.zeros(q.shape[0], self.n_dof, self.n_dof).to(q.device)
        for i in range(self.n_dof):
            grad_i = torch.autograd.grad(
                dL_dqdot[:, i].sum(), q_dot,
                create_graph=True,
                allow_unused=True,
            )[0]
            if grad_i is None:
                grad_i = torch.zeros_like(q_dot)
            M[:, i, :] = grad_i
        
        # ∂²L/∂q̇∂q * q̇
        corr = torch.zeros(q.shape[0], self.n_dof).to(q.device)
        for i in range(self.n_dof):
            grad_i = torch.autograd.grad(
                dL_dqdot[:, i].sum(), q,
                create_graph=True,
                allow_unused=True,
            )[0]
            if grad_i is None:
                grad_i = torch.zeros_like(q)
            corr[:, i] = (grad_i * q_dot).sum(dim=-1)
        
        # RHS = ∂L/∂q - corr
        rhs = dL_dq - corr  # shape (batch, n_dof)
        
        # Solve M * q̈ = rhs
        q_ddot = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
        
        return q_ddot
    
    def get_energy(
        self,
        q: torch.Tensor,
        q_dot: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get T, V, and total energy E = T + V.
        
        Returns:
            T, V, E
        """
        T = self.T_net(q_dot)
        V = self.V_net(q)
        E = T + V
        return T, V, E
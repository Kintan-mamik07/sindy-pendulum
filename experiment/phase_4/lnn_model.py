"""
Phase 4: Train Lagrangian Neural Network on single pendulum data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.insert(0, '/home/slippin-kd/sindy_pendulum')

from src.utils.data_generation import PendulumGenerator
from src.models.lnn import LagrangianNN

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("=" * 60)
print("PHASE 4: LNN TRAINING — SINGLE PENDULUM")
print("=" * 60)

# ── 1. Generate Data ─────────────────────────────────────
print("\n[1/5] Generating trajectory data...")

gen = PendulumGenerator(g=9.81, l=1.0, dt=0.01, t_max=10.0)
t   = np.arange(0, 10, 0.01)
y0_list = [
    [np.pi/6, 0.0],
    [np.pi/4, 0.0],
    [np.pi/3, 0.0],
    [np.pi/2, 0.0],
    [np.pi/3, 0.0],
]

data_list  = [gen.single_pendulum(t, y0, damping=0.0) for y0 in y0_list]
data_all=np.vstack(data_list)
q     = data_all[:, 0:1]   # θ
q_dot = data_all[:, 1:2]   # θ̇

q_ddot = np.gradient(q_dot[:, 0], 0.01).reshape(-1, 1)

print(f"  Trajectory shape: {data_all.shape}")
print(f"  q shape:          {q.shape}")
print(f"  q_dot shape:      {q_dot.shape}")
print(f"  q_ddot shape:     {q_ddot.shape}")

# ── True energy from physics ─────────────────────────────
# T = 0.5 * θ̇²   (l=1, m=1)
# V = -9.81 * cos(θ)
T_true_np = 0.5 * q_dot**2
V_true_np = -9.81 * np.cos(q)

# ── Convert to tensors ───────────────────────────────────
q_tensor      = torch.tensor(q,         dtype=torch.float32).to(device)
q_dot_tensor  = torch.tensor(q_dot,     dtype=torch.float32).to(device)
q_ddot_tensor = torch.tensor(q_ddot,    dtype=torch.float32).to(device)
T_true        = torch.tensor(T_true_np, dtype=torch.float32).to(device)
V_true        = torch.tensor(V_true_np, dtype=torch.float32).to(device)

# ── 2. Build Model ───────────────────────────────────────
print("\n[2/5] Building LNN...")

model   = LagrangianNN(n_dof=1, hidden_dim=64).to(device)
loss_fn = nn.MSELoss()

print(f"  KineticNet params:   {sum(p.numel() for p in model.T_net.parameters())}")
print(f"  PotentialNet params: {sum(p.numel() for p in model.V_net.parameters())}")
print(f"  Total params:        {sum(p.numel() for p in model.parameters())}")

# ── 3. Train T and V directly ────────────────────────────
print("\n[3/5] Training T and V networks directly...")

optimizer = optim.Adam(model.parameters(), lr=1e-3)
n_epochs  = 5000
log_every = 300

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    T_pred = model.T_net(q_dot_tensor)
    V_pred = model.V_net(q_tensor)

    loss_T = loss_fn(T_pred, T_true)
    loss_V = loss_fn(V_pred, V_true)
    loss   = loss_T + loss_V

    loss.backward()
    optimizer.step()

    if (epoch + 1) % log_every == 0:
        print(f"  Epoch [{epoch+1:>4}/{n_epochs}]  Loss: {loss.item():.6f}  T_loss: {loss_T.item():.6f}  V_loss: {loss_V.item():.6f}")

# ── 4. Validate ──────────────────────────────────────────
print("\n[4/5] Validation...")

model.eval()

with torch.no_grad():
    T_pred = model.T_net(q_dot_tensor)
    V_pred = model.V_net(q_tensor)
    E      = T_pred + V_pred
    E_np   = E.cpu().numpy()

    loss_T = loss_fn(T_pred, T_true).item()
    loss_V = loss_fn(V_pred, V_true).item()

print(f"  T MSE:        {loss_T:.6f}")
print(f"  V MSE:        {loss_V:.6f}")
print(f"  Energy mean:  {E_np.mean():.4f}")
print(f"  Energy std:   {E_np.std():.4f}  (lower = better conservation)")
print(f"  Energy range: [{E_np.min():.4f}, {E_np.max():.4f}]")

# True energy for comparison
E_true = T_true_np + V_true_np
print(f"\n  True Energy mean:  {E_true.mean():.4f}")
print(f"  True Energy std:   {E_true.std():.4f}")

# ── 5. Save ──────────────────────────────────────────────
print("\n[5/5] Saving model...")

save_path = '../../results/models/lnn_single_pendulum.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'n_dof':       1,
    'hidden_dim':  64,
    'loss_T':      loss_T,
    'loss_V':      loss_V,
    'epochs':      n_epochs,
}, save_path)

print(f"  ✓ Saved: {save_path}")

print("\n" + "=" * 60)
print("✓ Phase 4 LNN Training Complete")
print("=" * 60)
print("Next: run lnn_sindy.py to extract derivatives and run SINDy")
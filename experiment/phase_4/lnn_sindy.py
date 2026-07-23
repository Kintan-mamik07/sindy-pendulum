import numpy as np
import torch
import sys
sys.path.insert(0, '/home/slippin-kd/sindy_pendulum')

from src.utils.data_generation import PendulumGenerator
from src.models.lnn import LagrangianNN
from pysindy.feature_library import PolynomialLibrary, FourierLibrary, GeneralizedLibrary
from pysindy import SINDy
from pysindy.optimizers import STLSQ

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("=" * 60)
print("PHASE 4: LNN + SINDy HYBRID")
print("=" * 60)

# ── 1. Load Data ─────────────────────────────────────────
print("\n[1/5] Generating trajectory data...")

gen = PendulumGenerator(g=9.81, l=1.0, dt=0.01, t_max=10.0)
t   = np.arange(0, 10, 0.01)

# Single trajectory for SINDy
data_single  = gen.single_pendulum(t, [np.pi/3, 0.0], damping=0.0)
q_single     = data_single[:, 0:1]
q_dot_single = data_single[:, 1:2]

q_t_single     = torch.tensor(q_single,     dtype=torch.float32).to(device)
q_dot_t_single = torch.tensor(q_dot_single, dtype=torch.float32).to(device)

print(f"  Data shape: {data_single.shape}")

# ── 2. Load Trained LNN ──────────────────────────────────
print("\n[2/5] Loading trained LNN...")

checkpoint = torch.load('../../results/models/lnn_single_pendulum.pt', map_location=device)
model = LagrangianNN(n_dof=1, hidden_dim=64).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"  Loaded model trained for {checkpoint['epochs']} epochs")
print(f"  T MSE: {checkpoint['loss_T']:.6f}")
print(f"  V MSE: {checkpoint['loss_V']:.6f}")

# ── 3. Extract Clean Derivatives via Autograd ────────────
print("\n[3/5] Extracting smooth derivatives from LNN...")

q_t_s = q_t_single.requires_grad_(True)
V_s   = model.V_net(q_t_s)
dV_dq_s = torch.autograd.grad(V_s.sum(), q_t_s)[0]

# θ̈ = -∂V/∂θ  (mass matrix = 1 for m=1, l=1)
q_ddot_lnn = (-dV_dq_s).detach().cpu().numpy()

q_ddot_true = np.gradient(q_dot_single[:, 0], 0.01).reshape(-1, 1)

print(f"  ∂V/∂θ    range: [{dV_dq_s.detach().cpu().numpy().min():.3f}, {dV_dq_s.detach().cpu().numpy().max():.3f}]")
print(f"  θ̈ (LNN)  range: [{q_ddot_lnn.min():.3f}, {q_ddot_lnn.max():.3f}]")
print(f"  θ̈ (true) range: [{q_ddot_true.min():.3f}, {q_ddot_true.max():.3f}]")
print(f"\n  First 5 LNN accelerations:  {q_ddot_lnn[:5, 0]}")
print(f"  First 5 true accelerations: {q_ddot_true[:5, 0]}")

# ── 4. Run SINDy on LNN derivatives ─────────────────────
print("\n[4/5] Running SINDy on LNN-extracted derivatives...")

data_lnn  = np.hstack([q_single, q_dot_single])
x_dot_lnn = np.hstack([q_dot_single, q_ddot_lnn])

model_sindy = SINDy(
    feature_library=GeneralizedLibrary([
        PolynomialLibrary(degree=2),
        FourierLibrary(n_frequencies=1),
    ]),
    optimizer=STLSQ(threshold=0.5, alpha=1e-5),
)
model_sindy.fit(data_lnn, x_dot=x_dot_lnn, t=t)

print("\nDiscovered Equations (LNN + SINDy):")
for eq in model_sindy.equations():
    print(f"  {eq}")

coeffs   = model_sindy.coefficients()
n_active = np.count_nonzero(coeffs)
sparsity = 1.0 - (n_active / coeffs.size)
r2       = model_sindy.score(data_lnn, x_dot=x_dot_lnn, t=t)

print(f"\nMetrics:")
print(f"  R² Score:        {r2:.6f}")
print(f"  Active Features: {n_active} / {coeffs.size}")
print(f"  Sparsity:        {sparsity:.4f}")

# ── 5. Compare ───────────────────────────────────────────
print("\n[5/5] Comparison:")
print("-" * 60)
print("Ground Truth:        θ̈ = -9.81*sin(θ)")
print("Phase 1 (raw SINDy): θ̈ = -9.777*θ + 1.523*θ³")
print("Phase 4 (LNN+SINDy):")
for eq in model_sindy.equations():
    print(f"  {eq}")

print("\n" + "=" * 60)
print("✓ Phase 4 LNN + SINDy Complete")
print("=" * 60)
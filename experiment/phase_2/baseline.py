import numpy as np
import sys
sys.path.insert(0, '/home/slippin-kd/sindy_pendulum')
# UPDATE: Import generate_dataset alongside PendulumGenerator
from src.utils.data_generation import PendulumGenerator
from src.models.sindy_wrapper import SINDyWrapper
from src.visualize.plots import plot_trajectory, plot_phase_portrait

gen = PendulumGenerator(g=9.81, l=1.0, dt=0.01, t_max=10.0)
print("=" * 60)
print("PHASE 2: DOUBLE PENDULUM")
print("=" * 60)

# Step 1: Generate data
print("\n[1/4] Generating double pendulum trajectory...")
t = np.arange(0, 10, 0.01) #e
y0 = [np.pi/4, 0.0, np.pi/6, 0.0]  # θ₁=45°, θ̇₁=0, θ₂=30°, θ̇₂=0

data = gen.double_pendulum(t, y0, m1=1.0, m2=1.0, l1=1.0, l2=1.0)
print(f"  State shape: {data.shape}")
print(f"  θ₁ range: [{data[:, 0].min():.3f}, {data[:, 0].max():.3f}] rad")
print(f"  θ̇₁ range: [{data[:, 1].min():.3f}, {data[:, 1].max():.3f}] rad/s")
print(f"  θ₂ range: [{data[:, 2].min():.3f}, {data[:, 2].max():.3f}] rad")
print(f"  θ̇₂ range: [{data[:, 3].min():.3f}, {data[:, 3].max():.3f}] rad/s")

# Step 2: Plot
print("\n[2/4] Plotting trajectory...")
fig1 = plot_trajectory(t, data,
                       state_names=["θ₁ (rad)", "θ̇₁ (rad/s)", "θ₂ (rad)", "θ̇₂ (rad/s)"],
                       title="Double Pendulum: Trajectory")
fig1.savefig('../../results/figures/03_double_pendulum_trajectory.png', dpi=150)
print("  ✓ Saved: results/figures/03_double_pendulum_trajectory.png")

fig2 = plot_phase_portrait(data[:, :2], title="Double Pendulum: Phase Portrait (θ₁)")
fig2.savefig('../../results/figures/04_double_pendulum_phase.png', dpi=150)
print("  ✓ Saved: results/figures/04_double_pendulum_phase.png")

print("\n[3/4] Fitting SINDy (Polynomial Library, degree=3)...")

model = SINDyWrapper(
    feature_library='combined',  ## Use polynomial library for double pendulum
    optimizer='stlsq',  ## Use SINDyPI optimizer for better performance  
    threshold=0.5,
    degree=3
)
model.fit(data, t=t)

# Step 4: Results
print("\n[4/4] Results:")
print("-" * 60)
print("Discovered Equations:")
model.print_equations(precision=4)

print("\nMetrics:")
metrics = model.compute_metrics(data, t)
print(f"  R² Score:        {metrics['r2_score']:.6f}")
print(f"  Active Features: {metrics['n_active_features']} / {metrics['total_features']}")
print(f"  Sparsity:        {metrics['sparsity']:.4f}")

print("\n" + "=" * 60)
print("✓ Phase 2 Complete")
print("=" * 60)
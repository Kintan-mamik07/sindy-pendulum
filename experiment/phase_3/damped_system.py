

import numpy as np
import sys
sys.path.insert(0, '/home/slippin-kd/sindy_pendulum')

from src.utils.data_generation import PendulumGenerator
from src.models.sindy_wrapper import SINDyWrapper
from src.visualize.plots import plot_trajectory, plot_phase_portrait

gen = PendulumGenerator(g=9.81, l=1.0, dt=0.01, t_max=10.0)

print("=" * 60)
print("PHASE 3: DAMPED SYSTEMS")
print("=" * 60)

t = np.arange(0, 10, 0.01)
y0 = [np.pi/3, 0.0]
damping_coeffs = [0.0, 0.1, 0.5, 1.0]

results = {
    'damping': [],
    'r2_score': [],
    'n_active': [],
    'sparsity': [],
}

print("\nFitting SINDy at different damping levels:")
print("-" * 60)

for c in damping_coeffs:
    print(f"\nDamping Coefficient: c = {c:.1f}")
    print(f"  Ground Truth: θ̈ = -{c:.1f}*θ̇ - 9.81*sin(θ)")

    # Generate damped trajectory
    data = gen.single_pendulum(t, y0, damping=c)

    # Fit SINDy
    model = SINDyWrapper(
        feature_library='combined',
        optimizer='stlsq',
        threshold=0.5,
        degree=3,
    )
    model.fit(data, t=t)

    # Metrics
    metrics = model.compute_metrics(data, t)
    results['damping'].append(c)
    results['r2_score'].append(metrics['r2_score'])
    results['n_active'].append(metrics['n_active_features'])
    results['sparsity'].append(metrics['sparsity'])

    print(f"  R² Score:        {metrics['r2_score']:.4f}")
    print(f"  Active Features: {metrics['n_active_features']}")
    print(f"  Sparsity:        {metrics['sparsity']:.4f}")
    print("  Discovered Equations:")
    model.print_equations(precision=4)

    # Plot trajectory for each damping level
    fig = plot_trajectory(t, data,
                          state_names=["θ (rad)", "θ̇ (rad/s)"],
                          title=f"Damped Pendulum: c = {c:.1f}")
    fig.savefig(f'../../results/figures/phase3_damped_c{str(c).replace(".", "")}.png', dpi=150)
    print(f"  ✓ Saved: results/figures/phase3_damped_c{c}.png")

# Summary
print("\n" + "=" * 60)
print("Summary:")
print("-" * 60)
print(f"{'Damping (c)':<15} {'R² Score':<15} {'Active Feat':<15} {'Sparsity':<15}")
print("-" * 60)
for i, c in enumerate(results['damping']):
    print(f"{c:>13.1f}    {results['r2_score'][i]:>12.4f}    {results['n_active'][i]:>12}    {results['sparsity'][i]:>12.4f}")

print("\n✓ Phase 3 Complete")
print("=" * 60)
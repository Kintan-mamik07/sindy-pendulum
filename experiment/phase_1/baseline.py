import numpy as np
import sys
sys.path.insert(0, '/home/slippin-kd/sindy_pendulum')
# UPDATE: Import generate_dataset alongside PendulumGenerator
from src.utils.data_generation import PendulumGenerator, generate_dataset
from src.models.sindy_wrapper import SINDyWrapper
from src.visualize.plots import plot_trajectory, plot_phase_portrait

gen=PendulumGenerator(g=9.81, l=1.0, dt=0.01, t_max=10.0)

print("=" * 60)
print("PHASE 1: SINGLE NONLINEAR PENDULUM")
print("=" * 60)

print("\n[1/4] Generating noisy multi-trajectory data...")
# REPLACE the old single-trajectory generation with this:
# Generate 5 different trajectories, all with 5% noise
dataset = generate_dataset(system="single", noise_levels=[0.05], n_trajectories=5)

# Extract the list of 5 noisy arrays
train_data = dataset[0.05] 
t = np.linspace(0, 10, 500)

print(f"  Generated {len(train_data)} trajectories")
print(f"  State shape per trajectory: {train_data[0].shape}")
print(f"  θ range (Traj 0): [{train_data[0][:, 0].min():.3f}, {train_data[0][:, 0].max():.3f}] rad")
print(f"  θ̇ range (Traj 0): [{train_data[0][:, 1].min():.3f}, {train_data[0][:, 1].max():.3f}] rad/s")

#visualize the trajectory
print("\n[2/4] Plotting first trajectory...")

# UPDATE: Pass train_data[0] instead of data_clean so it plots the first noisy trajectory
fig1 = plot_trajectory(t, train_data[0], 
                       state_names=["θ (rad)", "θ̇ (rad/s)"],
                       title="Single Pendulum: Noisy Trajectory 0")
fig1.savefig('../../results/figures/01_clean_trajectory.png', dpi=150)
print("  ✓ Saved: results/figures/01_clean_trajectory.png")
 
fig2 = plot_phase_portrait(train_data[0], title="Single Pendulum: Phase Portrait (Noisy)")
fig2.savefig('../../results/figures/02_phase_portrait_clean.png', dpi=150)
print("  ✓ Saved: results/figures/02_phase_portrait_clean.png")

#Fit SINDy model

print("\n[3/4] Fitting SINDy (Polynomial Library, degree=3)...")

model=SINDyWrapper(
    feature_library='poly',
    optimizer="stlsq", #sequential thresholded least squares..assigning cooefficients to every single term in that library 
    threshold=0.05, #coefficients below this threshold will be set to zero
    degree=3, #upto degree 3 polynomial terms will be considered in the library..like sin(theta)=theta-theta^3/3!+theta^5/5!...
)

# UPDATE: Pass the list of trajectories and add the multiple_trajectories flag
model.fit(train_data, t=t)

#step 4: print the results

print("\n[4/4] Results:")
print("-" * 60)
print("Ground Truth Equation:")
print("  θ̇ = θ̇")
print("  θ̈ = -(g/l)*sin(θ)")
print("\nDiscovered Equations:")
model.print_equations(precision=4)

print("\nMetrics:")
# UPDATE: Pass train_data to the metrics calculator as well
# (If your custom wrapper throws an error here, you can change this back to train_data[0])
metrics = model.compute_metrics(train_data, t)
print(f"  R² Score: {metrics['r2_score']:.6f}")
# ... existing code ...
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


def plot_trajectory(
    t: np.ndarray,
    data: np.ndarray,
    state_names: list = None,
    title: str = "Pendulum Trajectory",
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    if state_names is None:
        state_names = [f"State {i}" for i in range(data.shape[1])]
    
    fig, axes = plt.subplots(data.shape[1], 1, figsize=figsize)
    if data.shape[1] == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(t, data[:, i], linewidth=2)
        ax.set_ylabel(state_names[i])
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_phase_portrait(
    data: np.ndarray,
    title: str = "Phase Portrait",
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(data[:, 0], data[:, 1], linewidth=1.5, alpha=0.7)
    ax.scatter(data[0, 0], data[0, 1], color="green", s=100, label="Start", zorder=5)
    ax.scatter(data[-1, 0], data[-1, 1], color="red", s=100, label="End", zorder=5)
    
    ax.set_xlabel("θ (radians)")
    ax.set_ylabel("θ̇ (rad/s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_noise_robustness(
    noise_levels: list,
    r2_scores: list,
    sparsity_values: list,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].plot(noise_levels, r2_scores, "o-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Noise Level")
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("Fit Quality vs Noise")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(noise_levels, sparsity_values, "s-", linewidth=2, markersize=8, color="orange")
    axes[1].set_xlabel("Noise Level")
    axes[1].set_ylabel("Sparsity")
    axes[1].set_title("Sparsity vs Noise")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
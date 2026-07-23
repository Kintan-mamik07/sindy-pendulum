# SINDy Nonlinear Pendulum Discovery

Sparse Identification of Nonlinear Dynamics (SINDy) applied to single and double pendulum systems, extended with a Lagrangian Neural Network (LNN) hybrid for exact equation discovery.

## Project Structure
sindy_pendulum/
├── config.yaml
├── data/
│ ├── processed/
│ └── raw/
├── experiment/
│ ├── phase_1/
│ │ └── baseline.py
│ ├── phase_2/
│ │ └── baseline.py
│ ├── phase_3/
│ │ └── damped_system.py
│ └── phase_4/
│ ├── lnn_train.py
│ └── lnn_sindy.py
├── results/
│ ├── figures/
│ └── models/
│ └── lnn_single_pendulum.pt
├── src/
│ ├── models/
│ │ ├── sindy_wrapper.py
│ │ └── lnn.py
│ ├── utils/
│ │ ├── data_generation.py
│ │ └── metrics.py
│ └── visualize/
│ └── plots.py
└── tests/

## Setup

### 1. Install

```bash
cd sindy_pendulum
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Experiments

```bash
# Phase 1: Single pendulum baseline
cd experiment/phase_1 && python baseline.py

# Phase 2: Double pendulum
cd experiment/phase_2 && python baseline.py

# Phase 3: Damped systems
cd experiment/phase_3 && python damped_system.py

# Phase 4: LNN + SINDy hybrid
cd experiment/phase_4
python lnn_train.py   # train LNN first
python lnn_sindy.py   # then run hybrid
```

---

## Experiments & Findings

### Phase 1: Single Nonlinear Pendulum ✅

**Goal:** Recover `θ̈ = -(g/l)*sin(θ)` from simulated trajectory data.

**Setup:**
- 1000 samples, `t ∈ [0, 10]`, `dt = 0.01`
- Initial condition: `θ₀ = π/3`, `θ̇₀ = 0`
- Polynomial library, degree=3, threshold=0.05

**Trajectory:**
- θ oscillates between -1.047 and +1.047 rad
- θ̇ oscillates between -1.0 and +1.0 rad/s
- Phase portrait shows closed orbit — confirms energy conservation

**SINDy Discovery:**
Ground Truth: θ̈ = -9.81sin(θ)
SINDy Discovered: θ̈ = -9.777θ + 1.523*θ³

| Metric | Value |
|--------|-------|
| R² Score | 1.000000 |
| Active Features | 3 / 20 |
| Sparsity | 0.8500 |

**Challenges:**
- Polynomial library approximates `sin(θ)` via Taylor expansion — not the exact form
- Initial attempt produced spurious `θ*θ̇²` term from numerical differentiation noise
- `FiniteDifference()` amplified noise in derivatives

**Improvements:**
- Switched from `FiniteDifference()` to `SmoothedFiniteDifference()` — eliminated spurious terms
- Fixed `data_generation.py` to include `g/l` factor in dynamics
- Used `np.arange` instead of `np.linspace` for exact `dt` consistency

---

### Phase 2: Double Pendulum ✅

**Goal:** Discover coupled 4D dynamics `[θ₁, θ̇₁, θ₂, θ̇₂]`.

**Setup:**
- 1000 samples, `dt = 0.01`, `m1=m2=l1=l2=1`
- Initial condition: `θ₁=π/4`, `θ̇₁=0`, `θ₂=π/6`, `θ̇₂=0`
- Polynomial library, degree=3, threshold=0.5

**SINDy Discovery:**
dθ₁/dt = θ̇₁
dθ̇₁/dt = -15.3θ₁ - 6.5θ₂ + cubic coupling terms
dθ₂/dt = θ̇₂
dθ̇₂/dt = 14.0θ₁ + 3.26θ₂ + cubic coupling terms

| Metric | Value |
|--------|-------|
| R² Score | 0.999752 |
| Active Features | 26 / 140 |
| Sparsity | 0.8143 |

**Challenges:**
- True equations have `sin(θ₁-θ₂)` and `cos(θ₁-θ₂)` coupling terms — polynomial library can't represent these exactly
- Combined library (Poly + Fourier) made things worse — 80 active features, overfitting
- Custom trig library applied sin/cos to velocity states — physically incorrect
- `SINDyPI` not available in installed pysindy version
- `SSR` optimizer produced 529 active features with zero sparsity

**Improvements:**
- Increased threshold from 0.05 to 0.5 — reduced active features from 47 to 26
- Fixed `dt` consistency using `np.arange(0, 10, 0.01)` instead of `np.linspace`
- Polynomial cross terms like `x0*x2²`, `x0²*x2` serve as approximations of true trig coupling

---

### Phase 3: Damped Systems ✅

**Goal:** Recover damping term `c*θ̇` in `θ̈ + c*θ̇ + (g/l)*sin(θ) = 0`.

**Setup:**
- Damping coefficients: `c ∈ {0.0, 0.1, 0.5, 1.0}`
- Combined library (Poly + Fourier), threshold=0.05

**Results:**

| Damping (c) | Discovered θ̈ | θ̇ recovered? | R² |
|-------------|--------------|--------------|-----|
| 0.0 | `-9.809*sin(θ)` | ✅ correctly absent | 1.0000 |
| 0.1 | `-14.543*sin(θ) + spurious` | ❌ too small | 0.9994 |
| 0.5 | `-9.548*sin(θ)` | ❌ absorbed | 0.9867 |
| 1.0 | `-1.000*θ̇ - 9.809*sin(θ)` | ✅ exact | 1.0000 |

**Key Observations:**
- `c=1.0` recovered perfectly — `-1.000*θ̇` exactly matches ground truth
- `c=0.1` and `c=0.5` damping terms too small relative to dominant `-9.81*sin(θ)` — thresholded out
- Combined library correctly discovered `sin(θ)` form (vs polynomial approximation in Phase 1)
- `c=0.0` discovered `-9.809*sin(θ)` — only 0.01% error from true `-9.81*sin(θ)`

**Challenge:**
- Small damping coefficients (c < 0.5) are fundamentally hard to recover when dominated by a much larger term

---

### Phase 4: LNN + SINDy Hybrid ✅

**Goal:** Use a Lagrangian Neural Network to learn clean energy landscape, then extract smooth derivatives for SINDy.

**Architecture:**
Raw trajectory [θ, θ̇]
↓
KineticNet(θ̇) → T = 0.5*||MLP(θ̇)||² (T > 0 enforced)
PotentialNet(θ) → V = MLP(θ)
↓
L = T - V (structured Lagrangian)
↓
∂V/∂θ via autograd → clean θ̈ = -∂V/∂θ
↓
SINDy discovers sparse equation from clean derivatives

**Training:**
- 5 trajectories with diverse initial conditions for better θ coverage
- 5000 epochs, Adam optimizer, lr=1e-3
- Direct supervision on T and V using known physics values

**Training Results:**
| Network | MSE |
|---------|-----|
| KineticNet (T) | 0.000005 |
| PotentialNet (V) | 0.002070 |

**SINDy Discovery:**
Ground Truth: θ̈ = -9.810sin(θ)
Phase 1 (raw SINDy): θ̈ = -9.777θ + 1.523θ³ (polynomial, 3 terms)
Phase 4 (LNN+SINDy): θ̈ = -9.842sin(θ) (exact form, 0.3% error)

| Metric | Phase 1 | Phase 4 |
|--------|---------|---------|
| R² Score | 1.000000 | 0.999991 |
| Active Features | 3 / 20 | 2 / 20 |
| Sparsity | 0.8500 | 0.9000 |
| Equation Form | Polynomial approx | Exact sin(θ) |

**Challenges:**
- Initial E-L forward pass during training caused loss of 464 — too expensive and unstable
- Device mismatch errors (CUDA vs CPU tensors) in autograd computation
- `∂²T/∂θ̇²` non-constant (0.151 to 1.119) — KineticNet didn't learn simple `0.5*θ̇²` structure cleanly
- Single trajectory gave poor θ coverage — LNN underestimated accelerations by 12%
- Gauge invariance — LNN can learn `L + dF/dt` which is physically equivalent but structurally different

**Improvements:**
- Switched from E-L training to direct T/V supervision using known physics values
- Added 5 diverse trajectories covering `θ ∈ [π/6, π/2]` for better coverage
- Used `∂V/∂θ` directly as acceleration instead of full E-L solve
- Used `GeneralizedLibrary([PolynomialLibrary, FourierLibrary])` to let SINDy find exact `sin(θ)`
- Split KineticNet and PotentialNet to enforce gauge invariance by design

---

## Final Comparison

| Phase | System | Method | Equation | R² | Sparsity |
|-------|--------|--------|----------|----|---------|
| 1 | Single | Raw SINDy (poly) | `-9.777*θ + 1.523*θ³` | 1.000 | 0.850 |
| 2 | Double | Raw SINDy (poly) | coupled polynomial | 0.9997 | 0.814 |
| 3 | Damped | SINDy (combined) | `-9.809*sin(θ) - 1.000*θ̇` | 1.000 | 0.932 |
| 4 | Single | LNN + SINDy | `-9.842*sin(θ)` | 0.9999 | 0.900 |

---

## Dependencies

- `numpy`, `scipy` — numerics
- `pysindy` — equation discovery
- `torch` — Lagrangian Neural Network
- `matplotlib`, `seaborn` — visualization
- `pytest` — testing

See `requirements.txt` for versions.

## Author

Kintan | GitHub: `Kintan-mamik07`
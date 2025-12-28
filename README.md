# **PhysiNet-Operator**

### **A Physics-Informed Neural Operator Framework for Spatiotemporal PDE Fields**

---

## Overview

**PhysiNet-Operator** is a research-grade framework for learning **operator mappings** in spatiotemporal physical systems using:

**Physics-informed neural operators**
**Probabilistic generative modeling**
**A synthetic PDE-based simulation gym built in JAX**

The framework focuses on modeling **partial differential equations (PDEs)** describing physical processes such as:

* wave propagation
* diffusion & transport
* reaction–diffusion patterns
* multiscale physical dynamics

through a **resolution-invariant neural operator** that supports:

* future state prediction
* trajectory generation
* uncertainty estimation

PhysiNet-Operator provides:

*  **PhysiGym** — a JAX-based differentiable PDE simulator
* **Multiscale neural operator architecture**
* **Probabilistic generative head**
* **Training, visualization & evaluation tools**

---

## Motivation

Modern ML is increasingly applied to **physical systems governed by PDEs**, including:

* fluid flow & wave systems
* advection–diffusion dynamics
* climate & geophysical modeling
* structural & material science simulation

However, PDE-based systems require:

* learning **function-to-function mappings**
* handling **multiscale & long-range interactions**
* providing **uncertainty-aware predictions**

**Neural operators** (FNO, DeepFDM, MINO, etc.) provide a powerful foundation for this domain.

**PhysiNet-Operator aims to unify:**

* probabilistic ML
* neural operator theory
* differentiable simulation
* structured physical modeling

into a **clean, extensible, research-oriented framework**.

---

## Key Contributions

PhysiNet-Operator introduces four main components:

### 1. PhysiGym — JAX PDE Simulation Environment

A fully-differentiable environment supporting:

* 2D Wave Equation
* 2D Heat Equation
* Advection–Diffusion
* Gray-Scott Reaction–Diffusion

Features include:

✔ randomized initial conditions & parameters
✔ multiresolution grids
✔ spectral / finite-difference solvers
✔ time-rollout generation
✔ export to `.npz`

---

### 2. Neural Operator Architecture

Hybrid design including:

* Multiscale Encoder
* **Fourier Neural Operator (FNO)** backbone
* Physics-informed residual blocks (DeepFDM-style)
* Probabilistic generative head
* Auxiliary PDE feature channels

Properties:

✨ resolution-invariant
✨ function-space modeling
✨ rollout-ready
✨ uncertainty-aware

---

### 3. Training Pipeline

Includes:

* PDE trajectory dataloaders
* deterministic & probabilistic objectives
* rollout consistency checks
* multiscale cropping
* HPC-friendly JAX/XLA execution

---

### 4. Visualization & Analysis Tools

Supports:

📌 PDE trajectory visualization
📌 multi-future sampling
📌 spectral analysis
📌 comparison vs classical solvers
📌 multiresolution benchmarking

---

## System Architecture

```
+---------------------------------------------------------------+
|                         PhysiNet-Operator                     |
+---------------------------------------------------------------+
|   1. PhysiGym (JAX PDE Simulator)                             |
|   2. Data Pipeline                                            |
|   3. Neural Operator Model                                    |
|   4. Training & Evaluation                                    |
+---------------------------------------------------------------+
```

---

##  Modules

### PhysiGym

Uniform JAX API:

```python
state = env.reset()
for t in range(T):
    state = env.step(state)
```

Supports:

* wave2d
* heat2d
* advection_diffusion2d
* gray_scott2d

Outputs include:

* field rollouts
* PDE coefficients
* boundary masks

---

### Multiscale Encoder

Extracts features across resolutions for **global-local modeling**.

---

### FNO Core

Frequency-domain operator learning via:

* spectral convolution
* truncated Fourier modes
* pointwise mixing

---

### ⚙ Physics-Informed Residual Blocks

Enforces stability & smoothness via:

```
u_pred ≈ u + dt * f(u)
```

---

### Generative Head (Optional)

Two options:

1. Gaussian mean + variance
2. Diffusion-based generator

Supports:

✨ multi-trajectory sampling
✨ uncertainty modeling

---

## Synthetic PDE Dataset

Example configuration:

* grid: 32×32 / 64×64
* timesteps: 32–128
* randomized ICs & coefficients

File structure:

```
{
  "u": [T, H, W],
  "params": {...},
  "boundary": [...]
}
```

---

## Training Pipeline

### Objective

Predict:

```
u(t+1) = G(u(t), conditioning)
```

Losses:

* L2
* relative error
* spectral loss
* PDE-residual regularization (optional)

Supports **autoregressive rollout training**.

Probabilistic mode adds:

* KL divergence
* ensemble variance analysis

---

## Experiments & Evaluation

✔ Single-step accuracy
✔ Multi-step stability
✔ Resolution generalization
✔ Parameter generalization
✔ Uncertainty calibration

---


## Research Impact

This project contributes to active research fields:

* neural operator learning
* physics-informed ML
* PDE modeling
* probabilistic physical simulation
* scalable JAX pipelines

It demonstrates:

⭐ theoretical grounding
⭐ engineering rigor
⭐ uncertainty-aware modeling

---

## Project Structure

```
physinet-operator/
│
├── physigym/                 # Synthetic PDE simulator (JAX)
│   ├── __init__.py
│   ├── configs.py            # Config dataclasses
│   ├── pde_wave.py           # 2D wave equation
│   ├── pde_heat.py           # 2D heat/diffusion equation
│   ├── pde_reacdiff.py       # Reaction–diffusion (Gray-Scott)
│   ├── env.py                # Unified gym-like interface + dataset generator
│   └── utils.py              # Grids, plotting hooks, helpers
│
├── physinet/                 # Neural operator models
│   ├── __init__.py
│   ├── encoder.py            # Multiscale encoder
│   ├── fno.py                # Fourier Neural Operator blocks
│   ├── head.py               # Deterministic + probabilistic heads
│   ├── model.py              # End-to-end PhysiNetOperator model
│   ├── data.py               # Data loader utilities
│   └── train.py              # Training loop
│
├── notebooks/                # Optional: exp notebooks
│   ├── 01_generate_data.ipynb
│   ├── 02_train_operator.ipynb
│   └── 03_visualize_rollouts.ipynb
│
├── data/
│   ├── synthetic/            # Generated trajectories (.npz)
│   └── external/             # Optional external PDE datasets
│
├── README.md
├── requirements.txt
└── pyproject.toml / setup.cfg (optional, kalau mau dipackage)
```

---

## 🔧 Installation (suggested)

```bash
git clone https://github.com/yourname/physinet-operator.git
cd physinet-operator
pip install -r requirements.txt
```

> Requires Python 3.10+ & JAX

---

## Contributions

PRs, suggestions, and research collaborations are welcome!

---

## License

MIT

---

## Acknowledgements

Inspired by work in:

* Neural Operators
* Physics-Informed Machine Learning
* Probabilistic Modeling

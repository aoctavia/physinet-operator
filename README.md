# MarineWave-Operator: WaveGym-JAX + Neural Operator for Marine Waves & Loads

MarineWave-Operator is a research prototype that combines a **JAX-based synthetic wave environment (WaveGym-JAX)** with a **neural operator model** for predicting wave elevation fields and wave-induced loads on a simplified ship hull.

The project is designed to bridge:
- **Probabilistic & generative ML** (neural operators, diffusion-style heads, uncertainty),
- **Computational physics** (PDE-based wave models),
- **Marine hydrodynamics & cybernetics** (wave kinematics, sea loads, operability).

This is especially aligned with current research directions in **maritime AI**, including:
- Neural operators in function spaces (e.g., FNO, DeepFDM, MINO),
- Multiscale, resolution-invariant modeling of wave fields,
- Synthetic training gyms based on numerical models using **JAX**,
- AI-based decision support under uncertain sea states.

---

## 1. Project Overview

We consider a simplified 2D sea surface around a ship. The core components are:

1. **WaveGym-JAX**
   - Differentiable synthetic environment using a simplified **wave PDE** (e.g., linear wave / shallow-water equation).
   - Generates:
     - Wave elevation fields over a spatial grid,
     - Time sequences of wave evolution,
     - Approximate **wave-induced loads** on an idealized hull.

2. **MarineWave-Operator Model**
   - A **neural operator** implemented in JAX/Flax:
     - **Multiscale encoder**: embeds spatial wave fields at different resolutions,
     - **Fourier Neural Operator (FNO)** core: models long-range wave propagation in function space,
     - (Optional) **physics-informed / DeepFDM blocks**: incorporate discrete PDE structure,
     - **Probabilistic head**: outputs mean + uncertainty (or samples) for future wave fields and loads.

3. **Training & Evaluation**
   - Train on synthetic data from WaveGym-JAX,
   - Evaluate:
     - Short- and mid-horizon wave field prediction,
     - Prediction of wave loads on the hull,
     - Calibration of predictive uncertainty.

---

## 2. Repository Structure

```text
MarineWave-Operator/
│
├── wavegym/
│   ├── pde.py                # Wave PDE definitions & numerical stepping (JAX)
│   ├── hull.py               # Simple hull model & load computation
│   ├── env.py                # WaveGym-JAX environment & data generation
│   └── utils.py              # Helper utilities (grids, plotting hooks, etc.)
│
├── mwo/
│   ├── fno.py                # Fourier Neural Operator (JAX/Flax)
│   ├── multiscale_encoder.py # Resolution-invariant encoder blocks
│   ├── head.py               # Predictive / probabilistic head for fields & loads
│   ├── operator_model.py     # Assembled MarineWave-Operator model
│   └── train.py              # Training loop / evaluation logic
│
├── notebooks/
│   ├── 01_generate_data.ipynb
│   ├── 02_train_operator.ipynb
│   ├── 03_visualize_rollouts.ipynb
│   ├── 04_ship_load_prediction.ipynb
│   └── 05_uncertainty_maps.ipynb
│
├── data/
│   ├── synthetic/            # Generated datasets
│   └── external/             # (Optional) Real/hindcast metocean data
│
├── requirements.txt
├── README.md
└── LICENSE

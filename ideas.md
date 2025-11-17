# 🌐 **PhysiNet-Operator**

### **A Physics-Informed Neural Operator Framework for Spatiotemporal PDE Fields**

---

## 📌 **1. Overview**

**PhysiNet-Operator** is a research-grade framework for learning **operator mappings** in spatiotemporal physical systems using **physics-informed neural operators**, **probabilistic generative modeling**, and a **synthetic PDE-based simulation gym built in JAX**.

The project focuses on modeling **partial differential equations (PDEs)** that describe physical processes—such as wave propagation, diffusion, transport, or pattern formation—through a **resolution-invariant neural operator** capable of predicting future states, generating new trajectories, and estimating uncertainty.

PhysiNet-Operator provides:

* a JAX-based PDE simulation engine (*PhysiGym*),
* a multiscale neural operator architecture,
* a probabilistic generative head for uncertainty estimation,
* tools for training, visualizing, and evaluating operator learning models.

---

## 🎯 **2. Motivation**

Modern machine learning is increasingly applied to **physical systems** governed by PDEs.
Examples include:

* fluid flow, waves, advection-diffusion,
* reaction–diffusion systems,
* structural dynamics,
* climate & geophysical models,
* large-scale scientific simulation.

Despite these advances, modeling PDE systems requires:

* learning **function-to-function mappings** rather than fixed-size tensors,
* handling **long-range interactions** and **multiscale structure**,
* producing **uncertainty-aware predictions**.

Neural operators—such as FNO, DeepFDM, MINO, and others—provide a powerful foundation for learning in these settings.

PhysiNet-Operator aims to build a **clean, extensible, and research-aligned** implementation that bridges:

* **probabilistic ML**,
* **neural operator theory**,
* **differentiable simulation**,
* **structured physical modeling**,
  into a single unified framework.

---

## 🚀 **3. Key Contributions**

PhysiNet-Operator introduces four core components:

### **1. PhysiGym (Synthetic PDE Environment in JAX)**

A fully differentiable environment for simulating:

* 2D Wave Equation
* 2D Heat Equation
* Advection-Diffusion
* Gray-Scott Reaction–Diffusion

Features:

* random initial conditions + PDE parameters
* multiresolution grid support
* spectral OR finite-difference solvers
* time rollout generation
* export to common ML formats (`.npz`)

---

### **2. Neural Operator Architecture**

A hybrid design combining:

* **Multiscale Encoder**
* **Fourier Neural Operator (FNO)** backbone
* **Physics-Informed Residual Blocks** (DeepFDM-style)
* **Probabilistic Generative Head** for uncertainty estimation
* **Auxiliary PDE feature channels** (boundary masks, coefficients)

Properties:

* resolution-invariant
* function-space modeling
* supports rollout prediction & trajectory generation
* uncertainty-aware (mean + variance or sampling)

---

### **3. Training Pipeline**

A modular training system including:

* data loading utilities for PDE trajectories
* deterministic or probabilistic loss functions
* rollout consistency checks
* multiscale random cropping
* operator learning objectives
* HPC-friendly training loops using JAX/XLA

---

### **4. Visualization + Analysis Tools**

Tools for:

* plotting PDE fields over time
* sampling multiple futures (uncertainty)
* spectral analysis of model representations
* comparison with classical PDE solvers
* multiresolution evaluation

---

## 🧩 **4. System Architecture**

```
+---------------------------------------------------------------+
|                         PhysiNet-Operator                    |
+---------------------------------------------------------------+
|                                                               |
|   1. PhysiGym (JAX PDE Simulator)                             |
|      - 2D wave, diffusion, advection-diffusion, Gray–Scott    |
|      - random ICs, PDE params                                 |
|      - rollout generation (T × H × W)                         |
|                                                               |
|   2. Data Pipeline                                             |
|      - Normalization                                           |
|      - Multiscale cropping                                     |
|      - Batch formation                                         |
|                                                               |
|   3. Neural Operator Model                                     |
|      +-------------------------------------------------------+ |
|      |  Multiscale Encoder  -->  FNO Core  -->  Gen. Head    | |
|      +-------------------------------------------------------+ |
|               |                          |                     |
|         future field              uncertainty map              |
|                                                               |
|   4. Training & Evaluation                                     |
|      - operator loss (L2, PDE residuals, KL)                  |
|      - rollout stability                                       |
|      - probabilistic sampling                                  |
|                                                               |
+---------------------------------------------------------------+
```

---

## 📦 **5. Modules (Detailed)**

### **A. PhysiGym**

A JAX-based PDE gym with consistent API:

```python
state = env.reset()
for t in range(T):
    state = env.step(state)
```

Supports PDE families:

* `wave2d`
* `heat2d`
* `advection_diffusion2d`
* `gray_scott2d`

Each generator yields:

* field evolution (T × H × W)
* PDE coefficients
* boundary masks (optional)

---

### **B. Multiscale Encoder**

Extracts features at multiple resolutions:

```
u(x,y) → downsampled pyramid → embeddings → combined representation
```

Benefits:

* captures short- and long-range structure
* helps operator generalization

---

### **C. FNO Core**

Implements frequency-domain operator learning:

* spectral convolution
* truncated Fourier modes
* local pointwise mixing
* skip connections

---

### **D. Physics-Informed Blocks**

Optional DeepFDM-style residual:

```
u_pred ≈ u + dt * f(u)          # f(u) learned to mimic PDE behavior
```

Helps enforce:

* stability
* physical structure
* smoother rollouts

---

### **E. Generative Head (Optional)**

Two variants:

1. **Mean + log-variance (Gaussian)**
2. **Diffusion-based stochastic generator**

   * produce multiple future trajectories
   * sample physical uncertainty

---

## 📊 **6. Synthetic PDE Dataset Design**

Dataset = collection of spatiotemporal fields generated by PhysiGym.

Example:

* grid sizes: 32×32 / 64×64
* T = 32–128 timesteps
* random initial fields (e.g. spectral noise, Gaussian bumps)
* random PDE coefficients
* optional noise injection

Output file format:

```
{
  "u": [T, H, W],
  "params": {...},
  "boundary": [...],
}
```

Multiple PDE families can be mixed to encourage generality.

---

## 🏋️ **7. Training Pipeline**

### (1) Operator Learning Objective

Predict:

```
u(t+1) = G(u(t), conditioning)
```

Loss:

* L2 per-pixel
* relative error
* spectral loss
* PDE-residual regularization (optional)

### (2) Rollout Training

Predict k steps autoregressively:

```
u(t+1)->u(t+2)-> ... -> u(t+k)
```

### (3) Probabilistic Objective (optional)

If using Generative Head:

* KL divergence
* sampling consistency
* trajectory ensemble variance

---

## 🧪 **8. Experiments & Evaluation**

### 1. Single-step prediction error

* MSE, MAE, PSNR

### 2. Multi-step rollout stability

Check if prediction diverges or stays stable.

### 3. Resolution generalization

Train at 64×64 → test at 128×128.

### 4. Parameter generalization

Sample PDE coefficients (e.g. wave speed `c`).

### 5. Uncertainty calibration

* compare ensemble mean with ground truth
* sharpness vs calibration

---

## 📅 **9. 6-Week Development Roadmap**

### **Week 1 — PhysiGym (PDE Simulator)**

* implement wave, heat, advection-diffusion
* random initial conditions
* dataset generator

### **Week 2 — Data Pipeline**

* normalization
* multiscale cropping
* batching

### **Week 3 — Multiscale Encoder + FNO**

* encoder
* spectral conv
* FNO block
* basic training loop

### **Week 4 — Physics-Informed Blocks**

* DeepFDM residual
* stability improvements

### **Week 5 — Generative Head**

* Gaussian head
* diffusion head (optional)
* uncertainty sampling

### **Week 6 — Experiments + Visualization**

* rollout tests
* resolution scaling
* uncertainty visualization
* final documentation

---

## 🎓 **10. Why This Project Has Strong Research Value**

This project touches active research themes:

* operator learning
* physics-informed ML
* PDE modeling
* generative modeling for physical systems
* uncertainty estimation
* JAX/XLA HPC pipelines

It demonstrates:

* modeling theory
* engineering execution
* understanding of physical structure
* probabilistic reasoning

---

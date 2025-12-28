# Statistical Physics Formulation for PDE Operator Learning

This document summarizes the **statistical physics and probabilistic formulation** underlying *PhysiNet-Operator*, focusing on **spatiotemporal PDE fields** and **neural operator learning**.

The goal is to connect:

* **PDE-governed dynamics**
* **probabilistic modeling**
* **operator learning in function space**
* **uncertainty estimation**

into a single mathematical framework.

---

## 1. PDE Fields as Random Functions

Let a physical field be:

[
u(x,t)\in \mathbb{R}, \quad x\in\Omega\subset\mathbb{R}^d,; t\ge 0
]

governed by a PDE:

[
\mathcal{F}\left(u,\frac{\partial u}{\partial t}, \nabla u,\nabla^2 u,\dots;,\theta\right)=0
]

where:

* (\mathcal{F}): PDE operator
* (\theta): physical parameters
* (\Omega): spatial domain

Examples:

| System              | PDE                                           |
| ------------------- | --------------------------------------------- |
| Heat                | (\partial_t u = D\nabla^2 u)                  |
| Wave                | (\partial_{tt} u = c^2\nabla^2 u)             |
| Advection-Diffusion | (\partial_t u + v\cdot\nabla u = D\nabla^2 u) |
| Gray-Scott          | Reaction-diffusion system                     |

---

## 2. Operator Learning Objective

We learn an operator mapping:

[
\mathcal{G}:; u_0(x);\mapsto;u(x,t)
]

or equivalently time-step evolution:

[
u_{t+1} = \mathcal{G}(u_t,\theta)
]

A neural operator approximator:

[
\hat{\mathcal{G}}_\phi \approx \mathcal{G}
]

where (\phi) are trainable parameters.

Unlike CNNs,
[
\hat{\mathcal{G}}_\phi:;\mathcal{H}\to\mathcal{H}
]
acts on **function spaces**, not finite tensors.

---

## 3. Probabilistic Formulation

We treat field evolution as **a stochastic process**:

[
p(u_{t+1}\mid u_t,\theta)
]

A deterministic model gives:

[
u_{t+1} = \hat{\mathcal{G}}_\phi(u_t)
]

A **probabilistic model** gives:

[
u_{t+1}\sim p_\phi(\cdot\mid u_t)
]

Examples:

* Gaussian output:

[
p_\phi(u_{t+1}\mid u_t)
=======================

\mathcal{N}
\left(
\mu_\phi(u_t),
\Sigma_\phi(u_t)
\right)
]

* diffusion-based sampler
* latent variable models

Uncertainty is encoded in:

✔ process noise
✔ parametric uncertainty
✔ epistemic uncertainty

---

## 4. Statistical Physics Viewpoint

### Field distribution

We define a probability functional:

[
P[u] \propto e^{-\beta \mathcal{H}[u]}
]

where:

* (\mathcal{H}[u]): Hamiltonian functional
* (\beta = (k_BT)^{-1})

For PDE systems, (\mathcal{H}) depends on gradients:

Example (Ginzburg–Landau-type):

[
\mathcal{H}[u]
==============

\int_\Omega
\left[
\frac{1}{2}
(\nabla u)^2
+
V(u)
\right]
dx
]

Expected physical observable:

[
\langle A[u]\rangle
===================

\int
\mathcal{D}u;
A[u],P[u]
]

This matches **Bayesian operator learning**.

---

## 5. Neural Operator as a Learned Transition Kernel

Neural operator models approximate:

[
p(u_{t+1}\mid u_t)
]

Rollout sequence:

[
p(u_{0:T})
==========

\prod_{t=0}^{T-1}
p(u_{t+1}\mid u_t)
]

Training objective:

[
\mathcal{L}
===========

* \sum_{t}
  \log
  p_\phi(u_{t+1}\mid u_t)
  ]

For Gaussian models:

[
\mathcal{L}
===========

\sum_t
\frac{|u_{t+1}-\mu_\phi(u_t)|^2}{2\sigma_\phi^2}
+
\log\sigma_\phi
]

---

## 6. Spectral Representation & FNO

A field is expressed in Fourier basis:

[
u(x)
====

\sum_k
\hat{u}_k
e^{ik\cdot x}
]

The neural operator applies **spectral convolution**:

[
\hat{u}'_k
==========

R_\phi(k),\hat{u}_k
]

where (R_\phi) is learned.

This mimics **Green’s function operators**.

Low-frequency truncation encodes:

✔ smoothness
✔ long-range propagation

---

## 7. Physics-Informed Residual Formulation

We model PDE dynamics like:

[
u_{t+1}
\approx
u_t
+
\Delta t;
f_\phi(u_t)
]

similar to numerical solvers:

[
\partial_t u
\approx
f_\phi(u)
]

This improves:

* stability
* physical consistency
* rollout smoothness

---

## 8. Loss Functions

### Data Loss

[
\mathcal{L}_{\text{data}}
=========================

|u_{pred}-u_{true}|_2^2
]

### Spectral Loss

[
\mathcal{L}_{\text{freq}}
=========================

|\hat{u}*{pred}-\hat{u}*{true}|_2^2
]

### PDE-Residual Loss (optional)

[
\mathcal{L}_{\text{PDE}}
========================

\left|
\mathcal{F}(u_{pred})
\right|_2^2
]

Total:

[
\mathcal{L}
===========

\mathcal{L}*{data}
+
\lambda*{freq}\mathcal{L}*{freq}
+
\lambda*{PDE}\mathcal{L}_{PDE}
]

---

## 9. Uncertainty Quantification

Variance of predictive distribution:

[
\mathrm{Var}(u)
===============

## \mathbb{E}[u^2]

\mathbb{E}[u]^2
]

Calibration metrics include:

* ensemble spread
* negative log-likelihood
* CRPS

---

## 10. Resolution-Invariant Property

Operator learning acts on function space:

[
u(x)\rightarrow v(x)
]

so models generalize across grids:

[
64^2 \rightarrow 128^2
]

This is critical for **scientific ML**.

---

## 🔬 11. Evaluation Criteria

### Single-Step Error

[
|u_{t+1}-\hat{u}_{t+1}|
]

### Multi-Step Stability

[
|u_{t+k}-\hat{u}_{t+k}|
]

### Spectral Energy Preservation

[
E(k)=|\hat{u}_k|^2
]

### Uncertainty Calibration

Compare:

* ensemble mean
* empirical variance

---

## 12. Relevance to PhysiNet-Operator

This framework supports:

✅ probabilistic neural operators
✅ resolution-invariant learning
✅ PDE-consistent rollouts
✅ scientific interpretability

and forms the **theoretical backbone** of the project.

---

## Suggested Reading

* Li et al. — Fourier Neural Operator
* Karniadakis — Physics-Informed ML
* Goldenfeld — Statistical Physics of Fields
* Rasmussen — Gaussian Processes
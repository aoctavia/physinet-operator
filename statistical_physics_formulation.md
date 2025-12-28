# 📘 Statistical Physics Formulation for PDE Operator Learning

This document summarizes the **statistical physics and probabilistic formulation** underlying *PhysiNet-Operator*, focusing on **spatiotemporal PDE fields** and **neural operator learning in function space**.

The aim is to connect:

* **PDE-governed dynamics**
* **probabilistic modeling**
* **neural operator learning**
* **uncertainty estimation**

into a unified mathematical framework.

---

## 1️⃣ PDE Fields as Random Functions

Consider a physical field

$$
u(x,t)\in \mathbb{R},
\quad
x\in\Omega\subset\mathbb{R}^d,;
t\ge 0
$$

governed by a PDE

$$
\mathcal{F}!\left(
u,
\frac{\partial u}{\partial t},
\nabla u,
\nabla^2 u,\dots;,
\theta
\right)=0
$$

where

* $\mathcal{F}$ = PDE operator
* $\theta$ = physical parameters
* $\Omega$ = domain

Examples:

| System              | PDE                                          |
| ------------------- | -------------------------------------------- |
| Heat                | $\partial_t u = D\nabla^2 u$                 |
| Wave                | $\partial_{tt}u = c^2\nabla^2u$              |
| Advection–Diffusion | $\partial_t u + v\cdot\nabla u = D\nabla^2u$ |
| Gray-Scott          | Reaction–diffusion                           |

---

## 2️⃣ Operator Learning Objective

We learn a **function-space operator**

$$
\mathcal{G}: u_0(x)\mapsto u(x,t)
$$

or, in discrete time:

$$
u_{t+1} = \mathcal{G}(u_t,\theta)
$$

A neural operator approximates

$$
\hat{\mathcal{G}}_\phi \approx \mathcal{G}
$$

with parameters $\phi$.

Unlike CNNs,

$$
\hat{\mathcal{G}}_\phi:\mathcal{H}\rightarrow\mathcal{H}
$$

acts on **infinite-dimensional spaces**
→ enabling **resolution-invariant prediction**.

---

## 3️⃣ Probabilistic Formulation

We treat field evolution as **a stochastic process**

$$
p(u_{t+1}\mid u_t,\theta)
$$

Deterministic operator:

$$
u_{t+1} = \hat{\mathcal{G}}_\phi(u_t)
$$

Probabilistic operator:

$$
u_{t+1}\sim p_\phi(\cdot\mid u_t)
$$

Gaussian case:

$$
p_\phi(u_{t+1}\mid u_t)
=======================

\mathcal{N}
!\Big(
\mu_\phi(u_t),
\Sigma_\phi(u_t)
\Big)
$$

This supports:

✔ process noise
✔ epistemic uncertainty
✔ parameter uncertainty

---

## 4️⃣ Statistical Physics Viewpoint

Define a probability distribution over fields

$$
P[u]
\propto
e^{-\beta\mathcal{H}[u]}
$$

where

* $\mathcal{H}[u]$ = Hamiltonian functional
* $\beta = (k_BT)^{-1}$

Example (Ginzburg–Landau-type)

$$
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
$$

Expectation of observable:

$$
\langle A[u]\rangle
===================

\int
\mathcal{D}u;
A[u],
P[u]
$$

This parallels **Bayesian learning over functions**.

---

## 5️⃣ Neural Operator as Transition Kernel

Neural operators approximate

$$
p(u_{t+1}\mid u_t)
$$

Joint distribution:

$$
p(u_{0:T})
==========

\prod_{t=0}^{T-1}
p(u_{t+1}\mid u_t)
$$

Training objective:

$$
\mathcal{L}
===========

*

\sum_t
\log
p_\phi(u_{t+1}\mid u_t)
$$

Gaussian NLL:

$$
\mathcal{L}
===========

\sum_t
\frac{
|u_{t+1}-\mu_\phi(u_t)|^2
}
{2\sigma_\phi^2}
+
\log\sigma_\phi
$$

---

## 6️⃣ Spectral Representation & FNO

Represent field in Fourier basis

$$
u(x)
====

\sum_k
\hat{u}_k
,e^{ik\cdot x}
$$

Neural operator applies

$$
\hat{u}'_k
==========

R_\phi(k),
\hat{u}_k
$$

where $R_\phi(k)$ is learned.

This mimics

➡ Green’s function operators
➡ long-range coupling

Low-frequency truncation enforces smoothness.

---

## 7️⃣ Physics-Informed Residual Modeling

We learn dynamics in residual form:

$$
u_{t+1}
\approx
u_t
+
\Delta t,
f_\phi(u_t)
$$

analogous to

$$
\partial_t u = f(u)
$$

This improves:

* stability
* physical coherence
* long-rollout behavior

---

## 8️⃣ Loss Functions

### Data loss

$$
\mathcal{L}_{data}
==================

|u_{pred}-u_{true}|_2^2
$$

### Spectral loss

$$
\mathcal{L}_{freq}
==================

|\hat{u}*{pred}-\hat{u}*{true}|_2^2
$$

### PDE residual loss

$$
\mathcal{L}_{PDE}
=================

|\mathcal{F}(u_{pred})|_2^2
$$

### Total loss

$$
\mathcal{L}
===========

\mathcal{L}*{data}
+
\lambda*{freq}\mathcal{L}*{freq}
+
\lambda*{PDE}\mathcal{L}_{PDE}
$$

---

## 9️⃣ Uncertainty Quantification

Predictive variance

$$
\mathrm{Var}(u)
===============

## \mathbb{E}[u^2]

\mathbb{E}[u]^2
$$

Metrics:

* negative log-likelihood
* ensemble calibration
* CRPS

---

## 🔟 Resolution-Invariant Property

Neural operators act on functions

$$
u(x)\rightarrow v(x)
$$

so models generalize across grids

$$
64^2 ;\rightarrow; 128^2
$$

This is critical for **scientific ML**.

---

## 1️⃣1️⃣ Evaluation Criteria

Single-step error

$$
|u_{t+1}-\hat{u}_{t+1}|
$$

Multi-step stability

$$
|u_{t+k}-\hat{u}_{t+k}|
$$

Spectral energy

$$
E(k)=|\hat{u}_k|^2
$$

Uncertainty calibration via ensemble spread.

---

## 1️⃣2️⃣ Relevance to PhysiNet-Operator

This framework supports:

✅ probabilistic neural operators
✅ physics-informed learning
✅ resolution-invariance
✅ scientific interpretability

and forms the **theoretical backbone** of the project.

---

## 📚 Suggested Reading

* Fourier Neural Operator — Li et al.
* Physics-Informed ML — Karniadakis
* Statistical Physics of Fields — Goldenfeld
* Gaussian Processes — Rasmussen

---
# physigym/wave2d_hos.py

from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class WaveParams:
    dt: float = 0.01
    g: float = 9.81
    depth: float = 50.0
    nx: int = 64
    ny: int = 64


def laplacian_periodic(u: jnp.ndarray) -> jnp.ndarray:
    up    = jnp.roll(u, -1, axis=0)
    down  = jnp.roll(u, +1, axis=0)
    left  = jnp.roll(u, -1, axis=1)
    right = jnp.roll(u, +1, axis=1)
    return up + down + left + right - 4.0 * u


def _wave_step_hos_impl(eta: jnp.ndarray, phi: jnp.ndarray, params: WaveParams):
    """
    Simplified HOS-like nonlinear free-surface model.
    """

    # Gradients
    dphi_dx = 0.5 * (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0))
    dphi_dy = 0.5 * (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1))

    deta_dx = 0.5 * (jnp.roll(eta, -1, axis=0) - jnp.roll(eta, 1, axis=0))
    deta_dy = 0.5 * (jnp.roll(eta, -1, axis=1) - jnp.roll(eta, 1, axis=1))

    # Free-surface evolution
    eta_t = - (dphi_dx * deta_dx + dphi_dy * deta_dy)

    # Bernoulli equation
    grad_phi_sq = dphi_dx**2 + dphi_dy**2
    phi_t = - params.g * eta - 0.5 * grad_phi_sq

    # Euler step
    eta_next = eta + params.dt * eta_t
    phi_next = phi + params.dt * phi_t

    return eta_next, phi_next


# ✔ JIT dengan static params
wave_step_hos = jax.jit(
    _wave_step_hos_impl,
    static_argnums=(2,),   # makes params static
)

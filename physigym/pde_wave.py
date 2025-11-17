# physigym/pde_wave.py

from __future__ import annotations
import jax
import jax.numpy as jnp

from .utils import laplacian_periodic


@jax.jit
def wave_step(
    eta: jnp.ndarray,
    eta_t: jnp.ndarray,
    c: float,
    dx: float,
    dy: float,
    dt: float,
):
    """
    2D linear wave equation:
        d^2 eta / dt^2 = c^2 ∇^2 eta
    """
    lap = laplacian_periodic(eta, dx, dy)
    eta_tt = (c ** 2) * lap

    eta_t_next = eta_t + dt * eta_tt
    eta_next = eta + dt * eta_t_next

    return eta_next, eta_t_next

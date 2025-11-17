# physigym/pde_heat.py

from __future__ import annotations

import jax
import jax.numpy as jnp

from .configs import HeatConfig
from .utils import laplacian_periodic


@jax.jit
def heat_step(u: jnp.ndarray, cfg: HeatConfig) -> jnp.ndarray:
    """
    Single time step for the 2D heat/diffusion equation:
        du/dt = κ ∇^2 u
    """
    dx = cfg.length_x / cfg.nx
    dy = cfg.length_y / cfg.ny
    lap = laplacian_periodic(u, dx, dy)
    u_next = u + cfg.dt * cfg.kappa * lap
    return u_next

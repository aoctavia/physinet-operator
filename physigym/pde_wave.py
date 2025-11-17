# physigym/pde_wave.py

from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp

from .configs import WaveConfig
from .utils import laplacian_periodic


@jax.jit
def wave_step(
    eta: jnp.ndarray,
    eta_t: jnp.ndarray,
    cfg: WaveConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single time step for a simple linear 2D wave equation:
        d^2 eta / dt^2 = c^2 ∇^2 eta
    using a velocity-form update on a periodic domain.
    """
    dx = cfg.length_x / cfg.nx
    dy = cfg.length_y / cfg.ny
    lap = laplacian_periodic(eta, dx, dy)
    eta_tt = (cfg.c ** 2) * lap

    eta_t_next = eta_t + cfg.dt * eta_tt
    eta_next = eta + cfg.dt * eta_t_next

    return eta_next, eta_t_next

# physigym/pde_reacdiff.py

from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp

from .configs import ReactionDiffusionConfig


@jax.jit
def gray_scott_step(
    u: jnp.ndarray,
    v: jnp.ndarray,
    cfg: ReactionDiffusionConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Gray–Scott reaction–diffusion system:
        du/dt = Du ∇^2 u - u v^2 + F (1 - u)
        dv/dt = Dv ∇^2 v + u v^2 - (F + k) v

    Uses periodic BCs via roll.
    """
    dx = cfg.length_x / cfg.nx
    dy = cfg.length_y / cfg.ny

    def laplacian(u: jnp.ndarray) -> jnp.ndarray:
        u_xx = (jnp.roll(u, -1, axis=0) - 2 * u + jnp.roll(u, 1, axis=0)) / (dx ** 2)
        u_yy = (jnp.roll(u, -1, axis=1) - 2 * u + jnp.roll(u, 1, axis=1)) / (dy ** 2)
        return u_xx + u_yy

    Lu = laplacian(u)
    Lv = laplacian(v)

    uvv = u * (v ** 2)

    du = cfg.Du * Lu - uvv + cfg.F * (1.0 - u)
    dv = cfg.Dv * Lv + uvv - (cfg.F + cfg.k) * v

    u_next = u + cfg.dt * du
    v_next = v + cfg.dt * dv

    return u_next, v_next

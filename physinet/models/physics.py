# physinet/model/physics.py

from __future__ import annotations
import jax.numpy as jnp


def laplacian_periodic(u: jnp.ndarray) -> jnp.ndarray:
    """
    2D Laplacian with periodic boundary conditions.
    u: [..., H, W]
    """
    up    = jnp.roll(u, -1, axis=-2)
    down  = jnp.roll(u, +1, axis=-2)
    left  = jnp.roll(u, -1, axis=-1)
    right = jnp.roll(u, +1, axis=-1)
    return up + down + left + right - 4.0 * u


def wave_residual(
    u_prev: jnp.ndarray,
    u_cur: jnp.ndarray,
    u_next_pred: jnp.ndarray,
    c: float,
    dt: float,
) -> jnp.ndarray:
    """
    Discrete wave equation residual:

        (u_{t+1} - 2u_t + u_{t-1}) / dt^2  =  c^2 ∇^2 u_t

    All inputs shape: [B, H, W]
    Returns scalar residual (mean squared).
    """
    u_prev = jnp.asarray(u_prev)
    u_cur = jnp.asarray(u_cur)
    u_next_pred = jnp.asarray(u_next_pred)

    c = jnp.asarray(c)
    dt = jnp.asarray(dt)

    lap = laplacian_periodic(u_cur)            # [B,H,W]
    lhs = (u_next_pred - 2.0 * u_cur + u_prev) / (dt ** 2)
    rhs = (c ** 2) * lap

    res = lhs - rhs
    return jnp.mean(res ** 2)

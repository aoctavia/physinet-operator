# physinet/model/physics.py

import jax.numpy as jnp


def laplacian_periodic(u):
    up = jnp.roll(u, -1, axis=-2)
    down = jnp.roll(u, 1, axis=-2)
    left = jnp.roll(u, -1, axis=-1)
    right = jnp.roll(u, 1, axis=-1)
    return up + down + left + right - 4 * u


def wave_residual(u_prev, u_cur, u_next, c=1.0, dt=0.1):
    lhs = (u_next - 2 * u_cur + u_prev) / (dt * dt)
    rhs = (c * c) * laplacian_periodic(u_cur)
    return jnp.mean((lhs - rhs) ** 2)

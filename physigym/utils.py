# physigym/utils.py

from __future__ import annotations

import jax.numpy as jnp


def make_grid(nx: int, ny: int, length_x: float = 1.0, length_y: float = 1.0):
    """Create a regular 2D grid (for visualisation / coordinates if needed)."""
    x = jnp.linspace(0.0, length_x, nx, endpoint=False)
    y = jnp.linspace(0.0, length_y, ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return X, Y


def laplacian_periodic(u: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Simple 2D Laplacian with periodic boundary conditions."""
    u_xx = (jnp.roll(u, -1, axis=0) - 2.0 * u + jnp.roll(u, 1, axis=0)) / (dx ** 2)
    u_yy = (jnp.roll(u, -1, axis=1) - 2.0 * u + jnp.roll(u, 1, axis=1)) / (dy ** 2)
    return u_xx + u_yy

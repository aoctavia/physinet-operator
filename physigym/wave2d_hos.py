from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class WaveParams:
    dt: float = 0.01
    c: float = 1.0


def laplacian_2d(u):
    """
    u: [H, W]
    """
    # periodic padding
    u_pad = jnp.pad(u, ((1, 1), (1, 1)), mode="wrap")
    return (
        u_pad[1:-1, 2:] + u_pad[1:-1, :-2] + u_pad[2:, 1:-1] + u_pad[:-2, 1:-1] - 4.0 * u
    )


@jax.jit
def wave_step(u, v, params: WaveParams):
    """
    Simple second-order wave equation:
    ∂u/∂t = v
    ∂v/∂t = c^2 ∇^2 u
    """
    du = v
    dv = (params.c ** 2) * laplacian_2d(u)
    u_next = u + params.dt * du
    v_next = v + params.dt * dv
    return u_next, v_next


def simulate_wave(u0, v0, params: WaveParams, steps: int):
    """
    u0, v0: [H, W]
    returns: [T, H, W] (trajectory of u)
    """
    def body(carry, _):
        u, v = carry
        u_next, v_next = wave_step(u, v, params)
        return (u_next, v_next), u_next

    (_, _), us = jax.lax.scan(body, (u0, v0), None, length=steps)
    return us  # [T,H,W]

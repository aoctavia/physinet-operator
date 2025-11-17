# physinet/fno.py

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class SpectralConv2d(nn.Module):
    modes_x: int
    modes_y: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [B, H, W, C]
        Apply spectral convolution on a subset of low-frequency modes.
        """
        B, H, W, C = x.shape

        x_ft = jnp.fft.rfftn(x, axes=(1, 2))  # [B, H, W_r, C]
        H_ft, W_ft = x_ft.shape[1], x_ft.shape[2]

        mx = min(self.modes_x, H_ft)
        my = min(self.modes_y, W_ft)

        w_real = self.param(
            "w_real",
            nn.initializers.lecun_normal(),
            (C, self.out_channels, mx, my),
        )
        w_imag = self.param(
            "w_imag",
            nn.initializers.lecun_normal(),
            (C, self.out_channels, mx, my),
        )
        w = w_real + 1j * w_imag

        out_ft = jnp.zeros((B, H_ft, W_ft, self.out_channels), dtype=jnp.complex64)

        x_ft_slice = x_ft[:, :mx, :my, :]  # [B, mx, my, C]
        x_ft_slice = jnp.transpose(x_ft_slice, (0, 3, 1, 2))  # [B, C, mx, my]

        xw = jnp.einsum("bcmn,conm->bonm", x_ft_slice, w)  # [B, Cout, mx, my]
        xw = jnp.transpose(xw, (0, 2, 3, 1))               # [B, mx, my, Cout]

        out_ft = out_ft.at[:, :mx, :my, :].set(xw)

        x_out = jnp.fft.irfftn(out_ft, s=(H, W), axes=(1, 2))
        return x_out


class FNO2d(nn.Module):
    modes_x: int = 12
    modes_y: int = 12
    width: int = 64
    depth: int = 4
    out_channels: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [B, H, W, C_in]
        """
        x = nn.Conv(self.width, (1, 1))(x)

        for _ in range(self.depth):
            residual = x
            spec = SpectralConv2d(self.modes_x, self.modes_y, self.width)(x)
            local = nn.Conv(self.width, (1, 1))(x)
            x = nn.gelu(spec + local)
            x = x + residual

        x = nn.Conv(self.out_channels, (1, 1))(x)
        return x

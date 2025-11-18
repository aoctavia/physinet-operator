# physinet/model/fno_jax.py

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


class SpectralConv2D(nn.Module):
    """
    2D Fourier layer (spectral convolution) ala FNO.

    x: [B, H, W, C_in]
    output: [B, H, W, C_out]
    """
    in_channels: int
    out_channels: int
    modes_x: int
    modes_y: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [B, H, W, C_in]
        """
        B, H, W, C_in = x.shape
        assert C_in == self.in_channels, (
            f"Input channels {C_in} != in_channels {self.in_channels}"
        )

        # 2D FFT (real → complex), only positive frequencies in last axis
        x_ft = jnp.fft.rfft2(x, axes=(1, 2))  # [B, H, W//2+1, C_in]
        _, H_ft, W_ft, _ = x_ft.shape

        mx = min(self.modes_x, H_ft)
        my = min(self.modes_y, W_ft)

        # Truncate low-frequency modes
        x_ft_trunc = x_ft[:, :mx, :my, :]  # [B, mx, my, C_in]

        # Complex weights for truncated region
        # Shape: [mx, my, C_in, C_out]
        weight = self.param(
            "weight",
            nn.initializers.normal(stddev=0.02),
            (mx, my, C_in, self.out_channels),
            jnp.complex64,
        )

        # Spectral convolution: sum over input channels
        # bxyc, xyco -> bxyo
        out_ft_trunc = jnp.einsum("bxyc,xyco->bxyo", x_ft_trunc, weight)

        # Re-embed into full spectral grid
        out_ft = jnp.zeros(
            (B, H_ft, W_ft, self.out_channels),
            dtype=jnp.complex64,
        )
        out_ft = out_ft.at[:, :mx, :my, :].set(out_ft_trunc)

        # Inverse FFT to go back to physical space
        out = jnp.fft.irfft2(out_ft, s=(H, W), axes=(1, 2))  # [B,H,W,C_out]
        return out


class FNOBlock2D(nn.Module):
    """
    One FNO block: spectral conv + linear skip + nonlinearity.
    """
    width: int
    modes_x: int
    modes_y: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [B, H, W, width]
        """
        B, H, W, C = x.shape
        assert C == self.width

        spec = SpectralConv2D(
            in_channels=self.width,
            out_channels=self.width,
            modes_x=self.modes_x,
            modes_y=self.modes_y,
        )(x)  # [B,H,W,width]

        # Pointwise linear projection in physical space
        lin = nn.Dense(self.width, name="w_phys")(x)  # [B,H,W,width]

        out = spec + lin
        out = nn.gelu(out)
        return out


class FNO2D(nn.Module):
    """
    Full FNO 2D network for PDE modeling.

    - Input : [B, H, W, in_channels]
    - Output: [B, H, W, out_channels]
    """
    in_channels: int = 1
    out_channels: int = 1
    width: int = 64
    depth: int = 4
    modes_x: int = 12
    modes_y: int = 12

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [B, H, W, in_channels]
        """
        B, H, W, C_in = x.shape
        assert C_in == self.in_channels, (
            f"Expected input channels {self.in_channels}, got {C_in}"
        )

        # Lift input to higher-dimensional feature space
        x = nn.Dense(self.width, name="lift")(x)  # [B,H,W,width]

        # FNO blocks
        for i in range(self.depth):
            x = FNOBlock2D(
                width=self.width,
                modes_x=self.modes_x,
                modes_y=self.modes_y,
                name=f"fno_block_{i}",
            )(x)  # [B,H,W,width]

        # Projection back to physical dimension
        x = nn.Dense(self.out_channels, name="proj")(x)  # [B,H,W,out_channels]
        return x

# physinet/head.py

from __future__ import annotations
from typing import Dict

import jax.numpy as jnp
from flax import linen as nn


class DeterministicHead(nn.Module):
    out_channels: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [B, H, W, C]
        Returns: [B, H, W, out_channels]
        """
        x = nn.Conv(self.out_channels, (1, 1))(x)
        return x


class GaussianFieldHead(nn.Module):
    out_channels: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        x: [B, H, W, C]
        Returns mean + logvar of a Gaussian field.
        """
        mean = nn.Conv(self.out_channels, (1, 1))(x)
        logvar = nn.Conv(self.out_channels, (1, 1))(x)
        return {"mean": mean, "logvar": logvar}

# physinet/encoder.py

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class MultiScaleEncoder(nn.Module):
    width: int = 32
    num_scales: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [B, H, W, C]
        Returns: [B, H, W, width] encoded feature map.
        """
        feats = []
        cur = x
        for _ in range(self.num_scales):
            cur = nn.Conv(self.width, (3, 3), padding="SAME")(cur)
            cur = nn.gelu(cur)
            feats.append(cur)
            cur = nn.avg_pool(cur, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        # upsample back to highest resolution via simple interpolation
        out = feats[0]
        for f in feats[1:]:
            f_up = jnp.repeat(jnp.repeat(f, 2, axis=1), 2, axis=2)  # naive upsample
            out = out + f_up[:, : out.shape[1], : out.shape[2], :]

        return out

# physinet/encoder.py

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


def upsample_to(x: jnp.ndarray, target_h: int, target_w: int) -> jnp.ndarray:
    """
    Simple nearest-neighbor upsampling to match target resolution.
    """
    h, w = x.shape[1], x.shape[2]
    scale_h = target_h // h
    scale_w = target_w // w
    x_up = jnp.repeat(x, scale_h, axis=1)
    x_up = jnp.repeat(x_up, scale_w, axis=2)
    return x_up[:, :target_h, :target_w, :]


class MultiScaleEncoder(nn.Module):
    width: int = 32
    num_scales: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x shape: [B, H, W, C]
        Output: [B, H, W, width]
        """
        H, W = x.shape[1], x.shape[2]

        feats = []
        cur = x

        # Downsampling pyramid
        for _ in range(self.num_scales):
            cur = nn.Conv(self.width, (3, 3), padding="SAME")(cur)
            cur = nn.gelu(cur)
            feats.append(cur)
            cur = nn.avg_pool(cur, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        # Fuse multiscale features
        fused = jnp.zeros_like(feats[0])  # same resolution as level 0

        for f in feats:
            h, w = f.shape[1], f.shape[2]

            # if not same resolution, upsample
            if (h != H) or (w != W):
                f = upsample_to(f, H, W)

            fused = fused + f

        return fused

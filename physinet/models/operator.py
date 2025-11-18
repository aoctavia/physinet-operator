# physinet/model/operator.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.image as jimage
from flax import linen as nn

# Asumsi: PhysiNetOperator sudah diexport di physinet/model/__init__.py
# sehingga bisa diimport dengan ". import PhysiNetOperator"
from . import PhysiNetOperator


class MultiScaleOperator(nn.Module):
    """
    Multiscale wrapper around PhysiNetOperator.

    - Input  : x [B, H, W, 1]
    - Output : dict dengan key "mean" [B, H, W, 1]
      (supaya kompatibel dengan train.py lama yang pakai outputs["mean"])
    """
    probabilistic: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        x: [B, H, W, 1]
        """
        B, H, W, C = x.shape

        # ---- Downsample to 1/2 and 1/4 resolution ----
        # pakai jax.image.resize (bilinear)
        H2, W2 = H // 2, W // 2
        H4, W4 = H // 4, W // 4

        x_1x = x
        x_2x = jimage.resize(x, (B, H2, W2, C), method="linear")
        x_4x = jimage.resize(x, (B, H4, W4, C), method="linear")

        # ---- Three operators at different scales ----
        op_1x = PhysiNetOperator(probabilistic=self.probabilistic, name="op_1x")
        op_2x = PhysiNetOperator(probabilistic=self.probabilistic, name="op_2x")
        op_4x = PhysiNetOperator(probabilistic=self.probabilistic, name="op_4x")

        out_1x = op_1x(x_1x)["mean"]            # [B,H,W,1]
        out_2x = op_2x(x_2x)["mean"]            # [B,H/2,W/2,1]
        out_4x = op_4x(x_4x)["mean"]            # [B,H/4,W/4,1]

        # ---- Upsample back to original resolution ----
        out_2x_up = jimage.resize(out_2x, (B, H, W, 1), method="linear")
        out_4x_up = jimage.resize(out_4x, (B, H, W, 1), method="linear")

        # ---- Fuse multiscale outputs ----
        fused = (out_1x + out_2x_up + out_4x_up) / 3.0

        # Kembalikan dict supaya interface sama dengan PhysiNetOperator
        return {"mean": fused}

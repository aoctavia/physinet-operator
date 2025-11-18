# physinet/model/operator.py

from __future__ import annotations
import jax
import jax.numpy as jnp
import jax.image as jimage
from flax import linen as nn

from .physi_operator import PhysiNetOperator


class MultiScaleOperator(nn.Module):
    """
    3-scale PhysiNet operator:
    - 1x
    - 1/2x
    - 1/4x
    """
    probabilistic: bool = False

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape

        x1 = x
        x2 = jimage.resize(x, (B, H // 2, W // 2, C), method="linear")
        x4 = jimage.resize(x, (B, H // 4, W // 4, C), method="linear")

        op1 = PhysiNetOperator(probabilistic=self.probabilistic, name="op1")
        op2 = PhysiNetOperator(probabilistic=self.probabilistic, name="op2")
        op4 = PhysiNetOperator(probabilistic=self.probabilistic, name="op4")

        y1 = op1(x1)["mean"]
        y2 = op2(x2)["mean"]
        y4 = op4(x4)["mean"]

        y2 = jimage.resize(y2, (B, H, W, 1), method="linear")
        y4 = jimage.resize(y4, (B, H, W, 1), method="linear")

        fused = (y1 + y2 + y4) / 3.0

        return {"mean": fused}

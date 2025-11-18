# physinet/model/physi_operator.py

from __future__ import annotations
import jax.numpy as jnp
from flax import linen as nn

from .fno_jax import FNO2D


class PhysiNetOperator(nn.Module):
    """
    Wrapper untuk FNO2D agar output berupa {"mean": ...}
    """
    in_channels: int = 1
    out_channels: int = 1
    width: int = 64
    depth: int = 4
    modes_x: int = 12
    modes_y: int = 12
    probabilistic: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):

        fno = FNO2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            width=self.width,
            depth=self.depth,
            modes_x=self.modes_x,
            modes_y=self.modes_y,
        )

        mean = fno(x)

        return {"mean": mean}

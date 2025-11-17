# physinet/model.py

from __future__ import annotations
from typing import Dict

import jax.numpy as jnp
from flax import linen as nn

from .encoder import MultiScaleEncoder
from .fno import FNO2d
from .head import DeterministicHead, GaussianFieldHead


class PhysiNetOperator(nn.Module):
    """
    End-to-end operator model:
      - input: recent PDE field frames (and optional conditioning)
      - output: next field (or distribution over fields)
    """
    modes_x: int = 12
    modes_y: int = 12
    width: int = 64
    depth: int = 4
    probabilistic: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        x: [B, H, W, C_in] (e.g., stacked past frames)
        """
        enc = MultiScaleEncoder(width=self.width)(x)

        fno_in = jnp.concatenate([x, enc], axis=-1)
        fno_out = FNO2d(
            modes_x=self.modes_x,
            modes_y=self.modes_y,
            width=self.width,
            depth=self.depth,
            out_channels=self.width,
        )(fno_in)

        if self.probabilistic:
            head_out = GaussianFieldHead(out_channels=1)(fno_out)
            return head_out  # {"mean": ..., "logvar": ...}
        else:
            eta_next = DeterministicHead(out_channels=1)(fno_out)
            return {"mean": eta_next}

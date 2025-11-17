# physigym/configs.py

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PDEConfig:
    nx: int = 64
    ny: int = 64
    dt: float = 0.01
    steps: int = 64
    length_x: float = 1.0
    length_y: float = 1.0


@dataclass
class WaveConfig(PDEConfig):
    c: float = 1.0  # wave speed


@dataclass
class HeatConfig(PDEConfig):
    kappa: float = 0.1  # diffusion coefficient


@dataclass
class ReactionDiffusionConfig(PDEConfig):
    Du: float = 0.16
    Dv: float = 0.08
    F: float = 0.035
    k: float = 0.065

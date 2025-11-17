# physinet/data.py

from __future__ import annotations
from typing import Tuple

import os
import glob

import numpy as np


def load_pde_dataset(
    data_dir: str,
    seq_len: int = 4,
    field_key: str = "u",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load trajectories from PhysiGym dataset and
    build (input_sequence, target) pairs.

    Returns:
        xs: [N, seq_len, H, W]
        ys: [N, H, W]
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    xs = []
    ys = []

    for p in paths:
        d = np.load(p)
        u = d[field_key]  # [T, H, W]
        T = u.shape[0]

        for t in range(seq_len, T - 1):
            xs.append(u[t - seq_len : t, :, :])
            ys.append(u[t + 1, :, :])

    xs = np.stack(xs, axis=0)
    ys = np.stack(ys, axis=0)
    return xs, ys

# # physigym/env.py

# from __future__ import annotations
# from dataclasses import dataclass, field
# from typing import Literal, Dict


# import os
# from pathlib import Path

# import jax
# import jax.numpy as jnp
# import numpy as np

# from .configs import WaveConfig, HeatConfig, ReactionDiffusionConfig
# from .pde_wave import wave_step
# from .pde_heat import heat_step
# from .pde_reacdiff import gray_scott_step


# PDEType = Literal["wave2d", "heat2d", "gray_scott"]


# @dataclass
# class EnvConfig:
#     pde_type: PDEType = "wave2d"
#     wave: WaveConfig = field(default_factory=WaveConfig)
#     heat: HeatConfig = field(default_factory=HeatConfig)
#     gray_scott: ReactionDiffusionConfig = field(default_factory=ReactionDiffusionConfig)



# class PhysiGym:
#     """
#     Synthetic PDE environment:
#       - samples random initial conditions
#       - rolls PDE in time
#       - returns full trajectories as numpy arrays
#     """

#     def __init__(self, cfg: EnvConfig, seed: int = 0):
#         self.cfg = cfg
#         self.key = jax.random.PRNGKey(seed)

#     def _sample_wave_ic(self) -> Dict[str, jnp.ndarray]:
#         cfg = self.cfg.wave
#         self.key, k1, k2 = jax.random.split(self.key, 3)
#         eta0 = 0.1 * jax.random.normal(k1, (cfg.nx, cfg.ny))
#         eta_t0 = 0.0 * jax.random.normal(k2, (cfg.nx, cfg.ny))
#         return {"eta": eta0, "eta_t": eta_t0}

#     def _sample_heat_ic(self) -> jnp.ndarray:
#         cfg = self.cfg.heat
#         self.key, k = jax.random.split(self.key)
#         u0 = 0.1 * jax.random.normal(k, (cfg.nx, cfg.ny))
#         return u0

#     def _sample_gray_scott_ic(self) -> Dict[str, jnp.ndarray]:
#         cfg = self.cfg.gray_scott
#         nx, ny = cfg.nx, cfg.ny
#         self.key, k1, k2 = jax.random.split(self.key, 3)
#         u = jnp.ones((nx, ny))
#         v = jnp.zeros((nx, ny))

#         # small square perturbation
#         r = nx // 10
#         x0, x1 = nx // 2 - r, nx // 2 + r
#         y0, y1 = ny // 2 - r, ny // 2 + r
#         u = u.at[x0:x1, y0:y1].set(0.50)
#         v = v.at[x0:x1, y0:y1].set(0.25)

#         # noise
#         u = u + 0.01 * jax.random.normal(k1, (nx, ny))
#         v = v + 0.01 * jax.random.normal(k2, (nx, ny))
#         return {"u": u, "v": v}

#     def roll_out(self) -> Dict[str, np.ndarray]:
#         """
#         Generate one trajectory for the chosen PDE type.
#         Returns dictionary with numpy arrays.
#         """
#         if self.cfg.pde_type == "wave2d":
#             return self._roll_out_wave()
#         elif self.cfg.pde_type == "heat2d":
#             return self._roll_out_heat()
#         elif self.cfg.pde_type == "gray_scott":
#             return self._roll_out_gray_scott()
#         else:
#             raise ValueError(f"Unknown pde_type: {self.cfg.pde_type}")

#     def _roll_out_wave(self) -> Dict[str, np.ndarray]:
#         cfg = self.cfg.wave
#         ic = self._sample_wave_ic()
#         eta, eta_t = ic["eta"], ic["eta_t"]

#         states = []
#         for _ in range(cfg.steps):
#             states.append(np.array(eta))

#             # PATCH: wave_step without cfg object
#             eta, eta_t = wave_step(
#                 eta,
#                 eta_t,
#                 cfg.c,
#                 cfg.length_x / cfg.nx,
#                 cfg.length_y / cfg.ny,
#                 cfg.dt,
#             )

#         return {"u": np.stack(states, axis=0)}


#     def _roll_out_heat(self) -> Dict[str, np.ndarray]:
#         cfg = self.cfg.heat
#         u = self._sample_heat_ic()
#         states = []
#         for _ in range(cfg.steps):
#             states.append(np.array(u))
#             u = heat_step(u, cfg)
#         return {"u": np.stack(states, axis=0)}

#     def _roll_out_gray_scott(self) -> Dict[str, np.ndarray]:
#         cfg = self.cfg.gray_scott
#         ic = self._sample_gray_scott_ic()
#         u, v = ic["u"], ic["v"]

#         states_u = []
#         states_v = []
#         for _ in range(cfg.steps):
#             states_u.append(np.array(u))
#             states_v.append(np.array(v))
#             u, v = gray_scott_step(u, v, cfg)

#         return {
#             "u": np.stack(states_u, axis=0),
#             "v": np.stack(states_v, axis=0),
#         }


# def generate_dataset(
#     output_dir: str,
#     num_trajectories: int = 100,
#     pde_type: PDEType = "wave2d",
#     seed: int = 0,
# ):
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     cfg = EnvConfig(pde_type=pde_type)
#     env = PhysiGym(cfg=cfg, seed=seed)

#     for i in range(num_trajectories):
#         traj = env.roll_out()
#         path = os.path.join(output_dir, f"{pde_type}_traj_{i:05d}.npz")
#         np.savez_compressed(path, **traj)
#         print(f"[PhysiGym] Saved {path}")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--output_dir", type=str, default="data/synthetic")
#     parser.add_argument("--num_trajectories", type=int, default=10)
#     parser.add_argument("--pde_type", type=str, default="wave2d")
#     args = parser.parse_args()

#     generate_dataset(
#         output_dir=args.output_dir,
#         num_trajectories=args.num_trajectories,
#         pde_type=args.pde_type,  # type: ignore
#     )
# physigym/env.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Dict

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Existing configs
from .configs import WaveConfig, HeatConfig, ReactionDiffusionConfig

# Existing PDE solvers
from .pde_wave import wave_step
from .pde_heat import heat_step
from .pde_reacdiff import gray_scott_step

# NEW: HOS-like synthetic wave
from .wave2d_hos import WaveParams, wave_step_hos

# NEW: ship response dynamics
from .ship_response import ShipResponseEnv


# -------------------------------------------------------------------------------------------------
# Extended PDE type
# -------------------------------------------------------------------------------------------------
PDEType = Literal[
    "wave2d",
    "heat2d",
    "gray_scott",
    "wave2d_hos",      # NEW
    "ship_response",   # NEW
]


# -------------------------------------------------------------------------------------------------
# Environment configuration
# -------------------------------------------------------------------------------------------------
@dataclass
class EnvConfig:
    pde_type: PDEType = "wave2d"

    wave: WaveConfig = field(default_factory=WaveConfig)
    heat: HeatConfig = field(default_factory=HeatConfig)
    gray_scott: ReactionDiffusionConfig = field(default_factory=ReactionDiffusionConfig)

    # NEW CONFIG
    hos_steps: int = 100
    ship_steps: int = 150
    ship_dt: float = 0.1


# -------------------------------------------------------------------------------------------------
# PhysiGym environment
# -------------------------------------------------------------------------------------------------
class PhysiGym:
    """
    Synthetic PDE environment:
      - Samples random initial conditions
      - Rolls PDE in time
      - Returns full trajectories as numpy arrays
    """

    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self.key = jax.random.PRNGKey(seed)

    # ----------------------------------------------------------
    # Initial Conditions Sampling
    # ----------------------------------------------------------

    def _sample_wave_ic(self) -> Dict[str, jnp.ndarray]:
        cfg = self.cfg.wave
        self.key, k1, k2 = jax.random.split(self.key, 3)
        eta0 = 0.1 * jax.random.normal(k1, (cfg.nx, cfg.ny))
        eta_t0 = 0.0 * jax.random.normal(k2, (cfg.nx, cfg.ny))
        return {"eta": eta0, "eta_t": eta_t0}

    def _sample_wave_hos_ic(self) -> Dict[str, jnp.ndarray]:
        """
        HOS-like initial condition.
        """
        cfg = self.cfg.wave
        self.key, k1, k2 = jax.random.split(self.key, 3)
        eta0 = 0.1 * jax.random.normal(k1, (cfg.nx, cfg.ny))
        phi0 = 0.1 * jax.random.normal(k2, (cfg.nx, cfg.ny))  # velocity potential
        return {"eta": eta0, "phi": phi0}

    def _sample_heat_ic(self) -> jnp.ndarray:
        cfg = self.cfg.heat
        self.key, k = jax.random.split(self.key)
        u0 = 0.1 * jax.random.normal(k, (cfg.nx, cfg.ny))
        return u0

    def _sample_gray_scott_ic(self) -> Dict[str, jnp.ndarray]:
        cfg = self.cfg.gray_scott
        nx, ny = cfg.nx, cfg.ny
        self.key, k1, k2 = jax.random.split(self.key, 3)

        # base fields
        u = jnp.ones((nx, ny))
        v = jnp.zeros((nx, ny))

        # patch
        r = nx // 10
        x0, x1 = nx // 2 - r, nx // 2 + r
        y0, y1 = ny // 2 - r, ny // 2 + r
        u = u.at[x0:x1, y0:y1].set(0.50)
        v = v.at[x0:x1, y0:y1].set(0.25)

        # noise
        u = u + 0.01 * jax.random.normal(k1, (nx, ny))
        v = v + 0.01 * jax.random.normal(k2, (nx, ny))
        return {"u": u, "v": v}

    # ----------------------------------------------------------
    # PDE Rollouts
    # ----------------------------------------------------------

    def roll_out(self) -> Dict[str, np.ndarray]:
        """
        Generate one trajectory.
        """
        p = self.cfg.pde_type

        if p == "wave2d":
            return self._roll_out_wave()

        elif p == "wave2d_hos":
            return self._roll_out_wave_hos()

        elif p == "heat2d":
            return self._roll_out_heat()

        elif p == "gray_scott":
            return self._roll_out_gray_scott()

        elif p == "ship_response":
            return self._roll_out_ship()

        else:
            raise ValueError(f"Unknown pde_type: {p}")

    # ----------------------------------------------------------
    # Standard wave2d
    # ----------------------------------------------------------
    def _roll_out_wave(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg.wave
        ic = self._sample_wave_ic()
        eta, eta_t = ic["eta"], ic["eta_t"]

        states = []
        dx = cfg.length_x / cfg.nx
        dy = cfg.length_y / cfg.ny

        for _ in range(cfg.steps):
            states.append(np.array(eta))
            eta, eta_t = wave_step(eta, eta_t, cfg.c, dx, dy, cfg.dt)

        return {"u": np.stack(states, axis=0)}

    # ----------------------------------------------------------
    # NEW: Higher-Order Spectral (HOS-like) wave
    # ----------------------------------------------------------
    def _roll_out_wave_hos(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg.wave
        ic = self._sample_wave_hos_ic()
        eta, phi = ic["eta"], ic["phi"]

        params = WaveParams(
            dt=cfg.dt,
            g=9.81,
            depth=50.0,
            nx=cfg.nx,
            ny=cfg.ny,
        )

        states = []
        for _ in range(self.cfg.hos_steps):
            states.append(np.array(eta))
            eta, phi = wave_step_hos(eta, phi, params)

        return {"u": np.stack(states, axis=0)}

    # ----------------------------------------------------------
    # Heat equation
    # ----------------------------------------------------------
    def _roll_out_heat(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg.heat
        u = self._sample_heat_ic()

        states = []
        for _ in range(cfg.steps):
            states.append(np.array(u))
            u = heat_step(u, cfg)

        return {"u": np.stack(states, axis=0)}

    # ----------------------------------------------------------
    # Gray–Scott reaction–diffusion
    # ----------------------------------------------------------
    def _roll_out_gray_scott(self) -> Dict[str, np.ndarray]:
        cfg = self.cfg.gray_scott
        ic = self._sample_gray_scott_ic()
        u, v = ic["u"], ic["v"]

        states_u, states_v = [], []

        for _ in range(cfg.steps):
            states_u.append(np.array(u))
            states_v.append(np.array(v))
            u, v = gray_scott_step(u, v, cfg)

        return {
            "u": np.stack(states_u, axis=0),
            "v": np.stack(states_v, axis=0),
        }

    # ----------------------------------------------------------
    # NEW: Ship dynamics (surge–heave–pitch)
    # ----------------------------------------------------------
    def _roll_out_ship(self) -> Dict[str, np.ndarray]:
        env = ShipResponseEnv(dt=self.cfg.ship_dt)
        env.reset()

        T = self.cfg.ship_steps

        # wave elevation forcing (synthetic)
        eta = np.sin(np.linspace(0, 6 * np.pi, T)) * 0.5

        states = []
        for t in range(T):
            state = env.step(float(eta[t]))   # state = [x, z, theta]
            states.append(state)

        return {"state": np.stack(states, axis=0), "eta": eta}


# -------------------------------------------------------------------------------------------------
# Dataset Generator
# -------------------------------------------------------------------------------------------------
def generate_dataset(
    output_dir: str,
    num_trajectories: int = 100,
    pde_type: PDEType = "wave2d",
    seed: int = 0,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig(pde_type=pde_type)
    env = PhysiGym(cfg=cfg, seed=seed)

    for i in range(num_trajectories):
        traj = env.roll_out()
        path = os.path.join(output_dir, f"{pde_type}_traj_{i:05d}.npz")
        np.savez_compressed(path, **traj)
        print(f"[PhysiGym] Saved {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/synthetic")
    parser.add_argument("--num_trajectories", type=int, default=10)
    parser.add_argument("--pde_type", type=str, default="wave2d")
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_trajectories=args.num_trajectories,
        pde_type=args.pde_type, 
    )

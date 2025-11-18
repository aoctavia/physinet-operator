import numpy as np


class ShipResponseEnv:
    """
    Very simplified surge-heave-pitch response to wave elevation eta(t).
    x: surge, z: heave, theta: pitch.
    """

    def __init__(self, dt=0.1):
        self.dt = dt
        # dummy coefficients
        self.kx, self.kz, self.kt = 0.5, 1.0, 0.2
        self.state = np.zeros(3, dtype=np.float32)  # [x, z, theta]

    def reset(self):
        self.state[:] = 0.0
        return self.state.copy()

    def step(self, eta_t: float):
        x, z, th = self.state

        # toy linear response
        dx = -self.kx * x + 0.1 * eta_t
        dz = -self.kz * z + 0.5 * eta_t
        dth = -self.kt * th + 0.05 * eta_t

        x_next = x + self.dt * dx
        z_next = z + self.dt * dz
        th_next = th + self.dt * dth

        self.state[:] = [x_next, z_next, th_next]
        return self.state.copy()

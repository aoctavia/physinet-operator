import numpy as np

class MarineWaveGym:
    """
    Synthetic training gym for marine wave fields.
    Generates phase-resolved 2D waves using a simple linear wave model
    (placeholder for HOS / higher-fidelity solver).
    """

    def __init__(self, nx=64, ny=64, nt=50, dt=0.1,
                 spectrum="jonswap", seed=42):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.dt = dt
        self.spectrum = spectrum
        self.rng = np.random.default_rng(seed)

    def sample_sea_state(self):
        """Randomize significant wave height, peak period, direction."""
        Hs = self.rng.uniform(0.5, 3.0)   # Significant wave height
        Tp = self.rng.uniform(4.0, 12.0)  # Peak period
        theta = self.rng.uniform(-np.pi/4, np.pi/4)  # wave direction
        return {"Hs": Hs, "Tp": Tp, "theta": theta}

    def simulate(self, condition):
        """
        Generate a synthetic wave elevation field η(x, y, t)
        using linear superposition of random Fourier modes.
        This mirrors real marine wave modeling behaviour.
        """
        Hs = condition["Hs"]
        Tp = condition["Tp"]
        theta = condition["theta"]

        x = np.linspace(0, 50, self.nx)
        y = np.linspace(0, 50, self.ny)
        t = np.linspace(0, self.nt * self.dt, self.nt)

        X, Y, T = np.meshgrid(x, y, t, indexing="ij")

        # random small frequency components
        kx = self.rng.uniform(0.1, 1.0)
        ky = self.rng.uniform(0.1, 1.0)
        omega = 2 * np.pi / Tp

        phase = self.rng.uniform(0, 2*np.pi)

        eta = (Hs / 2.0) * np.cos(
            kx * np.cos(theta) * X +
            ky * np.sin(theta) * Y -
            omega * T + phase
        )

        return eta  # shape [nx, ny, nt]

    def rollout_dataset(self, n_samples=10):
        """Generate many wave fields for training."""
        data = []
        for _ in range(n_samples):
            cond = self.sample_sea_state()
            eta = self.simulate(cond)
            data.append({"condition": cond, "eta": eta})
        return data

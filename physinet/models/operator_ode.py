import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint
except ImportError:
    raise ImportError(
        "Please install torchdiffeq: pip install torchdiffeq"
    )


class OperatorODE(nn.Module):
    """
    Wraps an operator model f(u) into du/dt = f(u) form for Neural ODE.
    """

    def __init__(self, operator: nn.Module):
        super().__init__()
        self.operator = operator

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # operator(u) should have same shape as u
        return self.operator(u)


def integrate_trajectory(
    ode_model: OperatorODE,
    u0: torch.Tensor,
    times: torch.Tensor,
    method: str = "rk4",
) -> torch.Tensor:
    """
    Integrate in continuous time.

    u0: [B, C, H, W]
    times: [T] time points (1D tensor)
    returns: [T, B, C, H, W]
    """
    traj = odeint(ode_model, u0, times, method=method)
    return traj  # shape [T, B, C, H, W]

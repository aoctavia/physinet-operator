import torch
import torch.nn.functional as F


def laplacian_2d(u: torch.Tensor) -> torch.Tensor:
    """
    2D Laplacian using finite differences with conv2d.
    u: [B, C, H, W]
    """
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=u.dtype,
        device=u.device,
    )
    kernel = kernel.view(1, 1, 3, 3)
    C = u.shape[1]
    kernel = kernel.repeat(C, 1, 1, 1)

    # padding=1 to keep same size
    lap = F.conv2d(u, kernel, padding=1, groups=C)
    return lap


def wave_pde_residual(
    u_prev: torch.Tensor,
    u_cur: torch.Tensor,
    u_next_pred: torch.Tensor,
    c: float,
    dt: float,
) -> torch.Tensor:
    """
    Discrete wave equation residual:
    (u_{t+1} - 2u_t + u_{t-1}) / dt^2 - c^2 ∇^2 u_t = 0
    """
    lap = laplacian_2d(u_cur)
    lhs = (u_next_pred - 2.0 * u_cur + u_prev) / (dt ** 2)
    rhs = (c ** 2) * lap
    res = lhs - rhs
    return (res ** 2).mean()

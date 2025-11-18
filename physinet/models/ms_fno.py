import torch
import torch.nn as nn
import torch.nn.functional as F

from .marine_fno import MarineFNO  # pastikan nama class sama dengan file kamu


class MultiscaleFNO(nn.Module):
    """
    Simple multiscale wrapper around MarineFNO.
    Uses three scales: 1x, 1/2x, 1/4x resolutions.
    """

    def __init__(self, modes: int = 12, width: int = 64):
        super().__init__()
        self.fno_1x = MarineFNO(modes=modes, width=width)
        self.fno_2x = MarineFNO(modes=modes, width=width)
        self.fno_4x = MarineFNO(modes=modes, width=width)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: [B, C, H, W] wave field
        returns: [B, C, H, W]
        """
        u1 = u
        u2 = F.avg_pool2d(u, kernel_size=2, stride=2)
        u3 = F.avg_pool2d(u, kernel_size=4, stride=4)

        o1 = self.fno_1x(u1)
        o2 = self.fno_2x(u2)
        o3 = self.fno_4x(u3)

        o2_up = F.interpolate(o2, size=u1.shape[-2:], mode="bilinear", align_corners=False)
        o3_up = F.interpolate(o3, size=u1.shape[-2:], mode="bilinear", align_corners=False)

        return o1 + o2_up + o3_up

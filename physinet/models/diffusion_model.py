import math
import torch
import torch.nn as nn

from .marine_fno import MarineFNO  # backbone for denoising UNet-like op


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] time step in [0,1] or [0,T]
        return: [B, dim]
        """
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -emb_factor)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class DiffusionWaveModel(nn.Module):
    """
    Denoising network for diffusion model on wave fields.
    Uses MarineFNO as spatial operator + time embedding.
    """

    def __init__(self, time_dim: int = 64, modes: int = 12, width: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # expect input: [B, C + time_dim_as_channel, H, W]
        self.backbone = MarineFNO(modes=modes, width=width, in_channels=1 + time_dim)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_noisy: [B, 1, H, W]
        t: [B] (float or int)
        returns: predicted noise, same shape as x_noisy
        """
        B, C, H, W = x_noisy.shape
        t_emb = self.time_embed(t)  # [B, time_dim]
        t_emb = t_emb.view(B, self.time_dim, 1, 1).expand(-1, -1, H, W)
        x_in = torch.cat([x_noisy, t_emb], dim=1)
        return self.backbone(x_in)

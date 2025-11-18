import torch
import torch.nn as nn


class SeaLoadHead(nn.Module):
    """
    Predicts global sea loads (Fx, Fy, Mz) from latent feature maps.
    Assumes input shape: [B, C, H, W].
    """

    def __init__(self, in_channels: int = 128, hidden: int = 64, out_dim: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Global average pooling over H, W
        # latent: [B, C, H, W]
        pooled = latent.mean(dim=[2, 3])  # [B, C]
        return self.mlp(pooled)  # [B, out_dim] = [B, Fx,Fy,Mz]
